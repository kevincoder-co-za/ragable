from django.db import models
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from django.http import JsonResponse
from django.views import View
import magic

from vectorapi.models import Document

@method_decorator(csrf_exempt, name="dispatch")
class DocumentUploadView(View):
    def post(self, request, *args, **kwargs):
        if not request.FILES.get("document_path"):
            return JsonResponse({"error": "No file uploaded"}, status=400)

        uploaded_file = request.FILES["document_path"]

        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(uploaded_file.read(1024))

        allowed_mime_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation", 
            "application/vnd.oasis.opendocument.text",
            "application/vnd.oasis.opendocument.presentation",
            "text/plain",
        ]

        if mime_type not in allowed_mime_types:
            return JsonResponse({"error": f"Unsupported file type: {mime_type}"}, status=400)

        document = Document.objects.create(
            team=request.user.ai_account,
            document_path=uploaded_file,
            status="pending",
        )

        return JsonResponse(
            {"message": "Document uploaded successfully!", "document_id": document.id},
            status=201
        )