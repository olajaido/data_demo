import boto3
import json
import requests
from urllib.parse import urljoin

class GrafanaDeployer:
    def __init__(self, workspace_id):
        self.workspace_id = workspace_id
        self.grafana = boto3.client('grafana')
        self.workspace_url = self.get_workspace_url()
        self.api_key = self.create_api_key()

    def get_workspace_url(self):
        response = self.grafana.describe_workspace(
            workspaceId=self.workspace_id
        )
        return response['workspace']['endpoint']

    def create_api_key(self):
        response = self.grafana.create_workspace_api_key(
            workspaceId=self.workspace_id,
            keyName='deployment-key',
            keyRole='ADMIN',
            secondsToLive=3600
        )
        return response['key']

    def deploy_dashboard(self, dashboard_path):
        with open(dashboard_path) as f:
            dashboard = json.load(f)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        url = urljoin(self.workspace_url, '/api/dashboards/db')
        payload = {
            'dashboard': dashboard,
            'overwrite': True
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Dashboard deployed: {response.json()['url']}")

if __name__ == '__main__':
    deployer = GrafanaDeployer('your-workspace-id')
    deployer.deploy_dashboard('model_monitoring.json')