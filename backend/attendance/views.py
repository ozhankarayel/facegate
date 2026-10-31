from datetime import timedelta

from django.conf import settings
from django.utils import timezone
from pgvector.django import CosineDistance
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from engine.service import bbox_area, decode_image, get_engine
from people.models import FaceTemplate

from .models import AccessLog
from .serializers import AccessLogSerializer


class RecognizeView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        file = request.FILES.get("image")
        if file is None:
            return Response({"detail": "'image' alanı zorunlu."}, status=400)
        try:
            img = decode_image(file.read())
        except ValueError:
            return Response({"detail": "Geçersiz görüntü."}, status=400)

        faces = get_engine().extract(img)
        if not faces:
            return Response({"status": "no_face"})

        # Birden fazla yüz varsa kameraya en yakın (en büyük) yüzü al
        face = max(faces, key=bbox_area)

        # pgvector: kosinüs uzaklığı, benzerlik = 1 - uzaklık
        match = (
            FaceTemplate.objects.select_related("person")
            .annotate(distance=CosineDistance("embedding", face["embedding"]))
            .order_by("distance")
            .first()
        )

        if match is None:
            return Response({"status": "empty_gallery"})

        score = 1.0 - float(match.distance)
        accepted = score >= settings.RECOGNITION_THRESHOLD

        result = AccessLog.Result.ACCEPTED if accepted else AccessLog.Result.REJECTED
        person = match.person if accepted else None

        # Cooldown: aynı kişi/sonuç için son 30 sn içinde log varsa yenisini yazma
        since = timezone.now() - timedelta(seconds=settings.LOG_COOLDOWN_SECONDS)
        already_logged = AccessLog.objects.filter(
            person=person, result=result, timestamp__gte=since
        ).exists()
        if not already_logged:
            AccessLog.objects.create(person=person, score=score, result=result)

        return Response(
            {
                "status": "accepted" if accepted else "rejected",
                "person": {"id": match.person.id, "name": match.person.name} if accepted else None,
                "score": round(score, 4),
                "bbox": face["bbox"],
            }
        )


class AccessLogListView(APIView):
    def get(self, request):
        logs = AccessLog.objects.select_related("person").order_by("-timestamp")[:100]
        return Response(AccessLogSerializer(logs, many=True).data)