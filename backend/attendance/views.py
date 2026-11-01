from datetime import timedelta

from django.conf import settings
from django.utils import timezone
from pgvector.django import CosineDistance
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from engine.liveness import EarExtractor, detect_blink
from .matching import identify
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

class VerifyView(APIView):
    """Çoklu kare: göz kırpma (liveness) + yüz tanıma."""
    parser_classes = [MultiPartParser]

    def post(self, request):
        files = request.FILES.getlist("images")
        if not (settings.LIVENESS_MIN_FRAMES <= len(files) <= 30):
            return Response(
                {"detail": f"{settings.LIVENESS_MIN_FRAMES}-30 arası kare gönderilmeli."},
                status=400,
            )
        try:
            frames = [decode_image(f.read()) for f in files]
        except ValueError:
            return Response({"detail": "Geçersiz görüntü."}, status=400)

        # 1) Liveness: her karede EAR, sonra kırpma örüntüsü
        with EarExtractor() as extractor:
            ears = [extractor.ear(f) for f in frames]
        liveness = detect_blink(ears, settings.LIVENESS_DROP_RATIO)

        candidates = [i for i, e in enumerate(ears) if e is not None]
        if not candidates:
            return Response({"status": "no_face", "liveness": liveness})

        # 2) Tanıma: gözlerin en açık olduğu 3 kare
        best = sorted(candidates, key=lambda i: ears[i], reverse=True)[:3]
        engine = get_engine()
        scores, person_ids, person_obj = [], set(), None
        for i in best:
            faces = engine.extract(frames[i])
            if not faces:
                continue
            face = max(faces, key=bbox_area)
            person, score = identify(face["embedding"])
            if person is None:
                return Response({"status": "empty_gallery"})
            scores.append(score)
            if score >= settings.RECOGNITION_THRESHOLD:
                person_ids.add(person.id)
                person_obj = person
            else:
                person_ids.add(None)

        if not scores:
            return Response({"status": "no_face", "liveness": liveness})

        mean_score = sum(scores) / len(scores)
        # Üç karede de aynı kişi eşleşmeli
        recognized = len(person_ids) == 1 and None not in person_ids

        # 3) Karar
        if not liveness["blink"]:
            result = AccessLog.Result.SPOOF
        elif recognized:
            result = AccessLog.Result.ACCEPTED
        else:
            result = AccessLog.Result.REJECTED

        log_person = person_obj if recognized else None
        since = timezone.now() - timedelta(seconds=settings.LOG_COOLDOWN_SECONDS)
        if not AccessLog.objects.filter(
            person=log_person, result=result, timestamp__gte=since
        ).exists():
            AccessLog.objects.create(person=log_person, score=mean_score, result=result)

        return Response(
            {
                "status": result,
                # Sahte şüphesinde kimlik bilgisi dışarı verilmez
                "person": {"id": person_obj.id, "name": person_obj.name}
                if result == AccessLog.Result.ACCEPTED
                else None,
                "score": round(mean_score, 4),
                "liveness": liveness,
            }
        )