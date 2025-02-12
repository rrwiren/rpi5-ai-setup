from google.oauth2 import service_account
from googleapiclient.discovery import build

def main():
    # 1. Provide the path to your service account JSON key
    SERVICE_ACCOUNT_FILE = "api-project-970767241608-5f275b1e55da.json"

    # 2. Scopes needed for Drive read/write. For read-only, you can also use:
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

    # 3. Create credentials object from the service account file
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, 
        scopes=SCOPES
    )

    # 4. Build the Drive API client
    service = build("drive", "v3", credentials=creds)

    # 5. Example: list files in the shared folder
    folder_id = "1aHrMTr3_nrLiN0sJ6_RTjr0NM230Pi3A"  # your RAG_FAISS folder ID
    query = f"'{folder_id}' in parents"

# 1aHrMTr3_nrLiN0sJ6_RTjr0NM230Pi3A


    results = service.files().list(q=query).execute()
    items = results.get("files", [])

    if not items:
        print("No files found.")
    else:
        print("Files in folder:")
        for item in items:
            print(f"{item['name']} ({item['id']})")

if __name__ == "__main__":
    main()
