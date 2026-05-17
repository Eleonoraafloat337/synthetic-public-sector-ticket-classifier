param namePrefix string
param location string
param keyVaultId string
param appInsightsId string

resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: replace('${namePrefix}mlstg', '-', '')
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: '${namePrefix}-aml'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: '${namePrefix}-aml'
    storageAccount: storage.id
    keyVault: keyVaultId
    applicationInsights: appInsightsId
  }
}

output workspaceName string = workspace.name
