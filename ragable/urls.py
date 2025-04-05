from django.urls import path
from vectorapi.api import DocumentUploadView

urlpatterns = [
    path("api/v1/documents/", DocumentUploadView.as_view(), name="add-document"),
]
