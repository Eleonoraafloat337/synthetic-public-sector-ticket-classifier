param namePrefix string
param location string
@allowed([
  'dev'
  'staging'
  'prod'
])
param environment string

var registryName = replace('${namePrefix}acr', '-', '')

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  sku: {
    name: environment == 'prod' ? 'Standard' : 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

output loginServer string = registry.properties.loginServer
