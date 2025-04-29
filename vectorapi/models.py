from django.db import models
from accounts.models import AiAccount

class Document(models.Model):
    ai_account = models.ForeignKey(AiAccount, on_delete=models.PROTECT)
    document_path = models.FileField(upload_to="documents/", blank=False, null=False)
    last_processed = models.DateTimeField()
    status = models.CharField(
        max_length=20,
        choices=(
            ("pending", "Pending"),
            ("complete", "Complete"),
            ("failed", "Failed"),
        )
    )
