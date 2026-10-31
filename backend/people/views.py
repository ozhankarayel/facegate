from django.conf import settings
from django.db import transaction
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from engine.service import decode_image, get_engine

from .models import FaceTemplate, Person
from .serializers import EnrollSerializer, PersonSerializer


class PersonListCreateView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request):
        people = Person.objects.order_by("-created_at")
        return Response(PersonSerializer(people, many=True).data)

    def post(self, request):
        serializer = EnrollSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        engine = get_engine()
        templates = []
        for i, f in enumerate(data["images"], start=1):
            try:
                img = decode_image(f.read())
            except ValueError:
                return Response({"detail": f"{i}. dosya geçerli bir görüntü değil."}, status=400)

            faces = engine.extract(img)
            if len(faces) != 1:
                return Response(
                    {"detail": f"{i}. fotoğrafta tam olarak 1 yüz olmalı (bulunan: {len(faces)})."},
                    status=400,
                )
            if faces[0]["score"] < settings.MIN_DET_SCORE:
                return Response({"detail": f"{i}. fotoğraftaki yüz yeterince net değil."}, status=400)
            templates.append(faces[0])

        # Ham fotoğraf saklanmaz; sadece embedding yazılır.
        with transaction.atomic():
            person = Person.objects.create(
                name=data["name"],
                email=data["email"],
                role=data["role"],
                consent_at=timezone.now(),
            )
            FaceTemplate.objects.bulk_create(
                [
                    FaceTemplate(person=person, embedding=t["embedding"], quality=t["score"])
                    for t in templates
                ]
            )
        return Response(PersonSerializer(person).data, status=status.HTTP_201_CREATED)


class PersonDetailView(APIView):
    def get(self, request, pk):
        return Response(PersonSerializer(get_object_or_404(Person, pk=pk)).data)

    def delete(self, request, pk):
        # Embedding'ler CASCADE ile silinir. Loglarda kişi alanı boşaltılır (SET_NULL).
        get_object_or_404(Person, pk=pk).delete()
        return Response(status=status.HTTP_204_NO_CONTENT)