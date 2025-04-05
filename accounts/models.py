import hashlib
import uuid

from cryptography.fernet import Fernet
from django.conf import settings
from django.db import models


class AiAccount(models.Model):
    # Because we want this to serve as an API Identifier as well.
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    name = models.CharField(max_length=100, blank=False, null=False)
    ai_api_key = models.BinaryField(editable=False)
    ai_chat_model_name = models.CharField(max_length=50, blank=False, null=False)
    ai_chat_embedding_model_name = models.CharField(
        max_length=50, blank=False, null=False
    )
    ai_embedding_dimensions = models.IntegerField(null=False, blank=False)
    ai_api_base_url = models.CharField(max_length=255, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    ragable_api_key = models.CharField(max_length=255, null=False, blank=False)

    @property
    def ai_api_key(self):
        f = Fernet(settings.SECRET_KEY)
        return f.decrypt(self.ai_api_key).decode()

    @ai_api_key.setter
    def ai_api_key(self, value):
        f = Fernet(settings.SECRET_KEY)
        self.ai_api_key = f.encrypt(value.encode())

    def get_namespace(self) -> str:
        return f"team_{self.pk}"
    
    def make_api_key(self):
        generated_key = uuid.uuid4()
        return (generated_key, AiAccount.get_api_key_hash(generated_key))

    @classmethod
    def get_api_key_hash(cls, api_key):
        return hashlib.sha256(api_key.encode()).hexdigest()