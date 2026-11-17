import sys
import os
from huggingface_hub import HfApi, create_repo, upload_file

token = sys.argv[1].strip()
repo_id = sys.argv[2].strip() if len(sys.argv) > 2 else None

api = HfApi(token=token)
user = api.whoami()
username = user['name']
print(f'Authenticated as Hugging Face user: {username}')

if not repo_id:
    repo_id = f'{username}/roadvision-ai-deeplabv3-resnet50'

print(f'Creating / accessing repository: {repo_id}')
repo_url = create_repo(repo_id=repo_id, repo_type='model', token=token, exist_ok=True)
print(f'Repository ready: {repo_url}')

model_path = r'D:\ROADVISION AI\artifacts\roadvision_model_artifacts\best_model.pth'
print(f'Uploading {model_path} to {repo_id}...')

upload_file(
    path_or_fileobj=model_path,
    path_in_repo='best_model.pth',
    repo_id=repo_id,
    repo_type='model',
    token=token
)
print('Upload complete!')
direct_url = f'https://huggingface.co/{repo_id}/resolve/main/best_model.pth'
print(f'DIRECT_DOWNLOAD_URL: {direct_url}')
