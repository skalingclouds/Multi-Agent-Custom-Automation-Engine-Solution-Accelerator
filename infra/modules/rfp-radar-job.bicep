// ========== rfp-radar-job.bicep ========== //
// Container Apps Job module for RFP Radar scheduled batch processing
// This module creates a scheduled Container Apps Job that runs the RFP Radar pipeline

@description('Required. The name of the Container Apps Job.')
param name string

@description('Optional. The location for the resource.')
param location string = resourceGroup().location

@description('Optional. Tags to apply to the resource.')
param tags object = {}

@description('Required. The resource ID of the Container App Environment.')
param environmentResourceId string

@description('Required. The resource ID of the user-assigned managed identity.')
param userAssignedIdentityResourceId string

@description('Required. The client ID of the user-assigned managed identity.')
param userAssignedIdentityClientId string

@description('Optional. The Container Registry hostname where the RFP Radar docker image is located.')
param containerRegistryHostname string = 'biabcontainerreg.azurecr.io'

@description('Optional. The Container Image Name to deploy.')
param containerImageName string = 'rfp-radar'

@description('Optional. The Container Image Tag to deploy.')
param containerImageTag string = 'latest'

// ========== Azure Service Endpoints ========== //

@description('Required. The Azure Storage Account blob endpoint URL.')
param storageAccountBlobUrl string

@description('Optional. The Azure Storage container name for RFP Radar.')
param storageContainerName string = 'rfp-radar'

@description('Required. The Azure AI Search service endpoint.')
param searchServiceEndpoint string

@description('Optional. The Azure AI Search index name for RFP Radar.')
param searchIndexName string = 'rfp-radar-index'

@description('Required. The Azure OpenAI service endpoint.')
param openAiEndpoint string

@description('Optional. The Azure OpenAI deployment name.')
param openAiDeploymentName string = 'gpt-4.1-mini'

@description('Optional. The Azure OpenAI API version.')
param openAiApiVersion string = '2024-12-01-preview'

// ========== Slack Configuration ========== //

@description('Required. The Key Vault secret URI for the Slack Bot Token.')
@secure()
param slackBotTokenSecretUri string

@description('Optional. The Slack channel to post digests to.')
param slackChannel string = '#bots'

// ========== RFP Radar Configuration ========== //

@description('Optional. The relevance threshold for RFP classification (0.0-1.0).')
param rfpRelevanceThreshold string = '0.55'

@description('Optional. Maximum age in days for RFPs to process.')
param rfpMaxAgeDays string = '3'

@description('Optional. The NAITIVE brand name for proposals.')
param naitiveBrandName string = 'NAITIVE'

@description('Optional. The NAITIVE website URL for proposals.')
param naitiveWebsite string = 'https://www.naitive.cloud'

// ========== Scheduling Configuration ========== //

@description('Optional. Cron expression for job schedule. Default is daily at 6 AM UTC.')
param cronExpression string = '0 6 * * *'

@description('Optional. Number of parallel job executions allowed.')
param parallelism int = 1

@description('Optional. Number of successful completions required.')
param replicaCompletionCount int = 1

@description('Optional. Maximum retry count for failed jobs.')
param replicaRetryLimit int = 3

@description('Optional. Job execution timeout in seconds. Default is 30 minutes.')
param replicaTimeout int = 1800

// ========== Monitoring Configuration ========== //

@description('Optional. The Application Insights connection string.')
param appInsightsConnectionString string = ''

@description('Optional. Enable/Disable usage telemetry for module.')
param enableTelemetry bool = true

// ========== Variables ========== //

var formattedUserAssignedIdentities = {
  '${userAssignedIdentityResourceId}': {}
}

// ========== Container Apps Job Resource ========== //

resource rfpRadarJob 'Microsoft.App/jobs@2024-03-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: formattedUserAssignedIdentities
  }
  properties: {
    environmentId: environmentResourceId
    workloadProfileName: 'Consumption'
    configuration: {
      triggerType: 'Schedule'
      scheduleTriggerConfig: {
        cronExpression: cronExpression
        parallelism: parallelism
        replicaCompletionCount: replicaCompletionCount
      }
      replicaRetryLimit: replicaRetryLimit
      replicaTimeout: replicaTimeout
      secrets: [
        {
          name: 'slack-bot-token'
          keyVaultUrl: slackBotTokenSecretUri
          identity: userAssignedIdentityResourceId
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'rfp-radar'
          image: '${containerRegistryHostname}/${containerImageName}:${containerImageTag}'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            // Application environment
            {
              name: 'APP_ENV'
              value: 'Prod'
            }
            // Azure identity
            {
              name: 'AZURE_TENANT_ID'
              value: tenant().tenantId
            }
            {
              name: 'AZURE_CLIENT_ID'
              value: userAssignedIdentityClientId
            }
            // Azure Storage configuration
            {
              name: 'AZURE_STORAGE_ACCOUNT_URL'
              value: storageAccountBlobUrl
            }
            {
              name: 'AZURE_STORAGE_CONTAINER'
              value: storageContainerName
            }
            // Azure AI Search configuration
            {
              name: 'AZURE_SEARCH_ENDPOINT'
              value: searchServiceEndpoint
            }
            {
              name: 'AZURE_SEARCH_INDEX_NAME'
              value: searchIndexName
            }
            // Azure OpenAI configuration
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: openAiEndpoint
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT'
              value: openAiDeploymentName
            }
            {
              name: 'AZURE_OPENAI_API_VERSION'
              value: openAiApiVersion
            }
            // Azure Cognitive Services scope for token-based auth
            {
              name: 'AZURE_COGNITIVE_SERVICES'
              value: 'https://cognitiveservices.azure.com/.default'
            }
            // Slack configuration
            {
              name: 'SLACK_BOT_TOKEN'
              secretRef: 'slack-bot-token'
            }
            {
              name: 'SLACK_CHANNEL'
              value: slackChannel
            }
            // RFP Radar configuration
            {
              name: 'RFP_RELEVANCE_THRESHOLD'
              value: rfpRelevanceThreshold
            }
            {
              name: 'RFP_MAX_AGE_DAYS'
              value: rfpMaxAgeDays
            }
            // NAITIVE branding
            {
              name: 'NAITIVE_BRAND_NAME'
              value: naitiveBrandName
            }
            {
              name: 'NAITIVE_WEBSITE'
              value: naitiveWebsite
            }
            // Application Insights (conditional)
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              value: appInsightsConnectionString
            }
          ]
        }
      ]
    }
  }
}

// ========== Telemetry ========== //

#disable-next-line no-deployments-resources
resource avmTelemetry 'Microsoft.Resources/deployments@2024-03-01' = if (enableTelemetry) {
  name: '46d3xbcp.ptn.rfp-radar-job.${replace('-..--..-', '.', '-')}.${substring(uniqueString(deployment().name, location), 0, 4)}'
  properties: {
    mode: 'Incremental'
    template: {
      '$schema': 'https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#'
      contentVersion: '1.0.0.0'
      resources: []
      outputs: {
        telemetry: {
          type: 'String'
          value: 'For more information, see https://aka.ms/avm/TelemetryInfo'
        }
      }
    }
  }
}

// ========== Outputs ========== //

@description('The name of the Container Apps Job.')
output name string = rfpRadarJob.name

@description('The resource ID of the Container Apps Job.')
output resourceId string = rfpRadarJob.id

@description('The resource group the Container Apps Job was deployed into.')
output resourceGroupName string = resourceGroup().name

@description('The location the resource was deployed into.')
output location string = rfpRadarJob.location
