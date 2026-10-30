from django.db import models
from pgvector.django import VectorField, HnswIndex


class Person(models.Model):
    name = models.CharField(max_length=120)
    email = models.EmailField(unique=True)
    role = models.CharField(max_length=40, default="student")
    consent_at = models.DateTimeField()          # açık rıza tarihi (KVKK)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class FaceTemplate(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name="templates")
    embedding = VectorField(dimensions=512)
    quality = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            HnswIndex(
                name="facetemplate_emb_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            )
        ]