from rest_framework import serializers

from .models import Person


class PersonSerializer(serializers.ModelSerializer):
    template_count = serializers.IntegerField(source="templates.count", read_only=True)

    class Meta:
        model = Person
        fields = ["id", "name", "email", "role", "consent_at", "created_at", "template_count"]


class EnrollSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=120)
    email = serializers.EmailField()
    role = serializers.CharField(max_length=40, required=False, default="student")
    consent = serializers.BooleanField()
    images = serializers.ListField(
        child=serializers.FileField(), min_length=1, max_length=10
    )

    def validate_email(self, value):
        if Person.objects.filter(email=value).exists():
            raise serializers.ValidationError("Bu e-posta zaten kayıtlı.")
        return value

    def validate_consent(self, value):
        if value is not True:
            raise serializers.ValidationError("Biyometrik veri için açık rıza zorunludur.")
        return value