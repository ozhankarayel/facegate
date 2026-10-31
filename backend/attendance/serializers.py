from rest_framework import serializers

from .models import AccessLog


class AccessLogSerializer(serializers.ModelSerializer):
    person_name = serializers.CharField(source="person.name", default=None, read_only=True)

    class Meta:
        model = AccessLog
        fields = ["id", "person", "person_name", "timestamp", "score", "result"]