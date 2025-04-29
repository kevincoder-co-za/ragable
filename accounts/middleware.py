from accounts.models import AiAccount
from django.http import JsonResponse

class AuthenticationMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        token = request.headers.get("Authorization")
        account_id = request.headers.get("AI_ACCOUNT_ID")

        if token is not None and account_id is not None:
            token = token.replace("Bearer ").strip()
            hash = AiAccount.get_api_key_hash(token)
            account = AiAccount.objects.filter(is_active=True, ragable_api_key=hash, pk=account_id)
            if account.exists():
                request.META['ai_account'] = account.first()
                return self.get_response(request)

        return JsonResponse(
            {"error": "Permission denied!"},
            status=403
        )
