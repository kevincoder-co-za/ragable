from django.core.management.base import BaseCommand
from accounts.models import AiAccount
import getpass

class Command(BaseCommand):
    help = 'Adds a new AI account to the API. This is basically credentials for a particular model.'

    def handle(self, *args, **options):
        # Prompt for input
        name = input("Enter a name that describes this account e.g. 'Developers' : ").strip()
        api_key = getpass.getpass("Please enter your API key: ").strip()
        model_name = input("Please enter the model name (e.g. gpt4o-mini): ").strip() or "gpt4o-mini"
        embedding_model = input("Embedding model name (e.g. text-embedding-ada-002): ").strip() or "text-embedding-3-small"
        base_url = input("API base URL (leave blank if using OpenAI defaults): ").strip() or None

        embedding_dims = None
        if embedding_model == "text-embedding-3-small":
            embedding_dims = 1536
        elif embedding_model == "text-embedding-3-large":
            embedding_dims = 3072
        
        if embedding_dims is None:
            embedding_dims = input(f"Sorry, cannot determine the dimensions of {embedding_model}. Please enter? e.g. 1536")

        # Create ai_account
        ai_account = AiAccount()
        ai_account.name = name
        ai_account.ai_chat_model_name = model_name
        ai_account.chat_embedding_model_name = embedding_model
        ai_account.ai_api_base_url = base_url
        ai_account.ai_api_key(api_key)
        ai_account.ai_embedding_dimensions = int(embedding_dims)
        ai_account.is_active = True

        (generated_key, generated_hash) = ai_account.make_api_key()
        ai_account.ragable_api_key = generated_hash
        
        if ai_account.save():
            self.stdout.write(self.style.SUCCESS(f"✅ ai_account created with ID: {ai_account.id}"))
            self.stdout.write(f"🔐 Namespace: {ai_account.get_namespace()}")
            self.stdout.write(f"🔐Ragable API Key: {generated_key}")
            return
        
        raise(f"Oops something went wrong! Failed to create ai_account: {name}")
