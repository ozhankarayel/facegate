from django.db import models
from people.models import Person


class AccessLog(models.Model):
    class Result(models.TextChoices):
        ACCEPTED = "accepted", "Kabul"
        REJECTED = "rejected", "Red"
        SPOOF = "spoof", "Sahte şüphesi"

    person = models.ForeignKey(Person, null=True, blank=True, on_delete=models.SET_NULL)
    timestamp = models.DateTimeField(auto_now_add=True)
    score = models.FloatField()
    result = models.CharField(max_length=10, choices=Result.choices)