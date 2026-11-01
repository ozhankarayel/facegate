from pgvector.django import CosineDistance

from people.models import FaceTemplate


def identify(embedding):
    """(person, benzerlik) döner. Galeri boşsa (None, None)."""
    match = (
        FaceTemplate.objects.select_related("person")
        .annotate(distance=CosineDistance("embedding", embedding))
        .order_by("distance")
        .first()
    )
    if match is None:
        return None, None
    return match.person, 1.0 - float(match.distance)