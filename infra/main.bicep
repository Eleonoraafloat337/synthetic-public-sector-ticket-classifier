@allowed([
  'dev'
  'staging'
  'prod'
])
param environment string
param location string = 'australiaeast'
param containerImageTag string

var namePrefix = 'pstc-${environment}'

module logs 'modules/log-analytics.bicep' = {
  name: 'logs'
  params: {
    namePrefix: namePrefix
    location: location
  }
}

module acr 'modules/container-registry.bicep' = {
  name: 'acr'
  params: {
    namePrefix: namePrefix
    location: location
    environment: environment
  }
}

module kv 'modules/key-vault.bicep' = {
  name: 'keyVault'
  params: {
    namePrefix: namePrefix
    location: location
  }
}

module aml 'modules/ml-workspace.bicep' = {
  name: 'aml'
  params: {
    namePrefix: namePrefix
    location: location
    keyVaultId: kv.outputs.keyVaultId
    appInsightsId: logs.outputs.appInsightsId
  }
}

module app 'modules/container-app.bicep' = {
  name: 'containerApp'
  params: {
    namePrefix: namePrefix
    location: location
    containerImage: '${acr.outputs.loginServer}/ticket-classifier:${containerImageTag}'
    logAnalyticsWorkspaceId: logs.outputs.workspaceId
    logAnalyticsCustomerId: logs.outputs.customerId
    logAnalyticsSharedKey: logs.outputs.sharedKey
    keyVaultUri: kv.outputs.keyVaultUri
  }
}

output containerAppUrl string = app.outputs.url
output acrLoginServer string = acr.outputs.loginServer
output mlWorkspaceName string = aml.outputs.workspaceName
