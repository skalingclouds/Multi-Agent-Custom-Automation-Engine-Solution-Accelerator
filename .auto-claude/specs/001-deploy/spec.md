# Specification: NAITIVE RFP Radar Deployment

## Overview

Deploy the NAITIVE RFP Radar system - an Azure-native automated RFP (Request for Proposal) discovery and proposal generation service. This system scrapes government RFP portals, filters opportunities by age and geography (US-only), uses Azure OpenAI to score relevance and extract tags, stores metadata and proposals in Azure Blob Storage and AI Search, generates Level 3 full proposals in markdown format, and delivers daily digests to a Slack channel (#bots). The goal is to create an "RFP scouting + proposal-writing team that never sleeps."

## Workflow Type

**Type**: feature

**Rationale**: This is a new feature deployment that adds a complete RFP Radar subsystem to the existing Multi-Agent Custom Automation Engine. It involves creating new Python modules, a new Docker container, infrastructure provisioning via Bicep, and integration with external services (Slack, government portals). The scope is significant but well-defined with clear deliverables.

## Task Scope

### Services Involved
- **rfp-radar** (primary) - New Python service for RFP scraping, classification, proposal generation, and Slack delivery
- **azure-infrastructure** (integration) - Bicep templates to provision Azure Container Apps Job, Blob Storage container, Search index
- **slack-integration** (integration) - Webhook/bot integration for delivering RFP digests

### This Task Will:
- [ ] Create new `src/rfp_radar/` Python module with orchestrator, scrapers, classifier, proposal generator, and digest builder
- [ ] Create Dockerfile for RFP Radar containerization
- [ ] Add Bicep modules for RFP Radar Azure infrastructure (Container Apps Job, dedicated blob container)
- [ ] Configure environment variables and secrets for RFP Radar
- [ ] Implement 3 scraper templates (GovTribe, OpenGov, BidNet patterns)
- [ ] Integrate with existing Azure OpenAI deployment for classification
- [ ] Configure Slack bot delivery to #bots channel
- [ ] Add deployment script for RFP Radar service

### Out of Scope:
- Production scraper endpoint configuration (placeholder URLs provided - real endpoints require portal-specific authentication)
- Slack workspace setup and bot creation (assumes pre-configured bot token)
- Azure subscription provisioning (leverages existing resource group)
- Custom portal scrapers beyond the 3 templates
- Frontend UI for RFP management

## Service Context

### RFP Radar Service (New)

**Tech Stack:**
- Language: Python 3.9+ (required by azure-storage-blob, pydantic v2, requests)
- Framework: Standalone (requests, azure-storage-blob, azure-search-documents, slack_sdk)
- Key directories: `src/rfp_radar/`, `src/rfp_radar/scrapers/`

**Entry Point:** `src/rfp_radar/main.py`

**How to Run:**
```bash
# Local development
python -m venv .venv
source .venv/bin/activate
pip install -r src/rfp_radar/requirements.txt
python src/rfp_radar/main.py

# Docker
docker build -t naitive-rfp-radar:latest -f src/rfp_radar/Dockerfile .
docker run --env-file .env naitive-rfp-radar:latest
```

**Port:** N/A (batch job, no exposed port)

### Existing Infrastructure

**Tech Stack:**
- Infrastructure: Bicep (Azure Resource Manager templates)
- CI/CD: GitHub Actions
- Container Registry: Azure Container Registry (biabcontainerreg.azurecr.io)

**Entry Point:** `infra/main.bicep`

**How to Deploy:**
```bash
az deployment group create \
  --resource-group <resource-group> \
  --template-file infra/main.bicep \
  --parameters location=eastus2
```

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `src/rfp_radar/main.py` | rfp-radar | Create main orchestrator (scrape → filter → classify → store → generate → notify) |
| `src/rfp_radar/config.py` | rfp-radar | Create config module loading all environment variables |
| `src/rfp_radar/classifier.py` | rfp-radar | Create AI relevance scoring using Azure OpenAI |
| `src/rfp_radar/proposal_generator.py` | rfp-radar | Create Level 3 proposal generation module |
| `src/rfp_radar/digest_builder.py` | rfp-radar | Create Slack message formatting module |
| `src/rfp_radar/models.py` | rfp-radar | Create Pydantic models for RFP data |
| `src/rfp_radar/storage_client.py` | rfp-radar | Create Azure Blob storage client |
| `src/rfp_radar/search_client.py` | rfp-radar | Create Azure AI Search client |
| `src/rfp_radar/slack_client.py` | rfp-radar | Create Slack SDK client wrapper |
| `src/rfp_radar/llm_client.py` | rfp-radar | Create Azure OpenAI REST client |
| `src/rfp_radar/logging_utils.py` | rfp-radar | Create structured logging utilities |
| `src/rfp_radar/scrapers/__init__.py` | rfp-radar | Create scraper registry and exports |
| `src/rfp_radar/scrapers/base.py` | rfp-radar | Create base scraper class with common interface |
| `src/rfp_radar/scrapers/govtribe.py` | rfp-radar | Create GovTribe-style aggregator scraper |
| `src/rfp_radar/scrapers/opengov.py` | rfp-radar | Create OpenGov portal scraper |
| `src/rfp_radar/scrapers/bidnet.py` | rfp-radar | Create BidNet portal scraper |
| `src/rfp_radar/Dockerfile` | rfp-radar | Create container image definition |
| `src/rfp_radar/requirements.txt` | rfp-radar | Create Python dependencies list |
| `infra/modules/rfp-radar-job.bicep` | infrastructure | Create Container Apps Job for scheduled execution |
| `.github/workflows/rfp-radar-deploy.yml` | ci/cd | Create deployment workflow for RFP Radar |
| `src/rfp_radar/tests/__init__.py` | rfp-radar | Create test package init |
| `src/rfp_radar/tests/test_config.py` | rfp-radar | Create config unit tests |
| `src/rfp_radar/tests/test_classifier.py` | rfp-radar | Create classifier unit tests |
| `src/rfp_radar/tests/test_proposal_generator.py` | rfp-radar | Create proposal generator unit tests |
| `src/rfp_radar/tests/test_digest_builder.py` | rfp-radar | Create digest builder unit tests |
| `src/rfp_radar/tests/test_models.py` | rfp-radar | Create Pydantic model unit tests |
| `src/rfp_radar/tests/test_scrapers.py` | rfp-radar | Create scraper interface unit tests |

## Files to Reference

These files show patterns to follow:

| File | Pattern to Copy |
|------|----------------|
| `src/backend/common/config/app_config.py` | Environment variable loading pattern, Azure credential handling |
| `src/backend/Dockerfile` | Docker build pattern for Python services |
| `infra/main.bicep` | Azure resource naming conventions, user-assigned identity pattern |
| `src/mcp_server/services/hr_service.py` | Service class pattern with async methods |
| `src/backend/common/utils/utils_date.py` | Date utility functions pattern |

## Patterns to Follow

### Environment Configuration Pattern

From `src/backend/common/config/app_config.py`:

```python
class AppConfig:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Required vs optional pattern
        self.REQUIRED_VAR = self._get_required("VAR_NAME")
        self.OPTIONAL_VAR = self._get_optional("VAR_NAME", "default")

    def _get_required(self, name: str, default: Optional[str] = None) -> str:
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"Environment variable {name} not found")

    def _get_optional(self, name: str, default: str = "") -> str:
        return os.environ.get(name, default)
```

**Key Points:**
- Use `_get_required()` for mandatory configuration
- Use `_get_optional()` for optional configuration with defaults
- Load via `python-dotenv` for local development
- Cache credentials and clients

### Azure Credential Pattern

From `src/backend/common/config/app_config.py`:

```python
def get_azure_credential(self, client_id=None):
    if self.APP_ENV == "dev":
        return DefaultAzureCredential()
    else:
        return ManagedIdentityCredential(client_id=client_id)
```

**Key Points:**
- Use `DefaultAzureCredential` for local development
- Use `ManagedIdentityCredential` for production
- Pass `client_id` for user-assigned managed identity

### Bicep Resource Naming Pattern

From `infra/main.bicep`:

```bicep
var solutionSuffix = toLower(trim(replace('${solutionName}${solutionUniqueText}', '-', '')))
var storageAccountName = replace('st${solutionSuffix}', '-', '')
var containerAppResourceName = 'ca-${solutionSuffix}'
```

**Key Points:**
- Use `solutionSuffix` for unique resource names
- Follow Azure naming conventions (prefixes: st, ca, cae, srch, kv)
- Remove special characters for storage accounts

## Requirements

### Functional Requirements

1. **RFP Portal Scraping**
   - Description: Scrape multiple government RFP portals using extensible scraper framework
   - Acceptance: Successfully fetch RFP listings from at least one portal template

2. **Age & Geography Filtering**
   - Description: Filter RFPs to US-only, max 3 days old
   - Acceptance: RFPs older than `RFP_MAX_AGE_DAYS` or non-US are excluded from processing

3. **AI Relevance Classification**
   - Description: Use Azure OpenAI to score RFP relevance (0-1) and extract tags (AI/Dynamics/Modernization)
   - Acceptance: Each RFP receives a relevance score; scores below `RFP_RELEVANCE_THRESHOLD` (0.55) are excluded

4. **Azure Storage Integration**
   - Description: Store RFP PDFs and metadata in Azure Blob Storage
   - Acceptance: RFP documents stored in `rfp-radar` container with proper metadata

5. **Azure AI Search Indexing**
   - Description: Index RFP metadata in Azure AI Search for discovery
   - Acceptance: RFPs searchable via Azure AI Search index `rfp-radar-index`

6. **Level 3 Proposal Generation**
   - Description: Generate full markdown proposals for relevant RFPs
   - Acceptance: Proposals stored in Blob Storage with NAITIVE branding

7. **Slack Digest Delivery**
   - Description: Post daily digest to Slack #bots channel with RFP summaries, scores, and links
   - Acceptance: Slack message posted with proper formatting, links to proposals

### Edge Cases

1. **Empty Scrape Results** - Log info message, skip classification/storage, send "no new RFPs" digest
2. **API Rate Limiting** - Implement exponential backoff with max 3 retries
3. **Azure OpenAI Timeout** - 60-second timeout with retry; skip RFP on persistent failure
4. **Malformed RFP Data** - Validate with Pydantic; log and skip invalid entries
5. **Slack API Failure** - Retry 3 times; log error but don't fail entire job
6. **Large PDF Files** - Azure Blob handles up to 64 MiB single upload; implement chunking for larger files

## Implementation Notes

### DO
- Follow the config pattern in `src/backend/common/config/app_config.py` for environment variables
- Reuse Azure credential pattern with `DefaultAzureCredential` (dev) and `ManagedIdentityCredential` (prod)
- Use Pydantic for data validation (consistent with existing codebase)
- Implement structured logging with `logging_utils.py`
- Use async where possible for I/O-bound operations
- Store proposals as markdown in Blob Storage with `.md` extension
- Include NAITIVE branding (`NAITIVE_BRAND_NAME`, `NAITIVE_WEBSITE`) in proposals

### DON'T
- Don't use OpenAI SDK directly - use REST API via requests (per requirements)
- Don't hardcode API endpoints or credentials
- Don't commit `.env` files or secrets to repository
- Don't create new Azure resources manually - use Bicep templates
- Don't block on Slack failures - log and continue
- Don't store sensitive data in Search index (use Blob for full documents)

## Development Environment

### Start Services

```bash
# Clone and setup
cd /Users/chris/ai/Multi-Agent-Custom-Automation-Engine-Solution-Accelerator

# Create virtual environment for RFP Radar
python -m venv .venv-rfp
source .venv-rfp/bin/activate

# Install dependencies
pip install -r src/rfp_radar/requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your Azure/Slack credentials

# Run locally
python src/rfp_radar/main.py

# Or via Docker
docker build -t naitive-rfp-radar:latest -f src/rfp_radar/Dockerfile .
docker run --env-file .env naitive-rfp-radar:latest
```

### Service URLs
- Azure Portal: https://portal.azure.com
- Azure Blob Storage: `https://<storage-account>.blob.core.windows.net/rfp-radar`
- Azure AI Search: `https://<search-service>.search.windows.net`
- Slack: https://app.slack.com (channel: #bots)

### Required Environment Variables

```bash
# Azure Storage
AZURE_STORAGE_ACCOUNT_URL="https://<storage-account>.blob.core.windows.net"
AZURE_STORAGE_CONTAINER="rfp-radar"
AZURE_STORAGE_SAS_TOKEN="?sv=..." # For dev; production uses Managed Identity

# Azure AI Search
AZURE_SEARCH_ENDPOINT="https://<search-service>.search.windows.net"
AZURE_SEARCH_API_KEY="<api-key>" # For dev; production uses Managed Identity
AZURE_SEARCH_INDEX_NAME="rfp-radar-index"

# Azure OpenAI
AZURE_OPENAI_ENDPOINT="https://<openai-resource>.openai.azure.com"
AZURE_OPENAI_API_KEY="<api-key>" # For dev; production uses Managed Identity
AZURE_OPENAI_DEPLOYMENT="gpt-4o"  # Or use custom deployment name (e.g., gpt-4.1 if configured)

# Slack
SLACK_BOT_TOKEN="xoxb-..."
SLACK_CHANNEL="#bots"

# RFP Radar Configuration
RFP_RELEVANCE_THRESHOLD="0.55"
RFP_MAX_AGE_DAYS="3"
NAITIVE_BRAND_NAME="NAITIVE"
NAITIVE_WEBSITE="https://www.naitive.cloud"

# Application Environment
APP_ENV="dev"  # or "prod"
AZURE_CLIENT_ID="<managed-identity-client-id>"  # For production
```

## Success Criteria

The task is complete when:

1. [ ] All Python modules in `src/rfp_radar/` are created and pass linting (Flake8)
2. [ ] Dockerfile builds successfully
3. [ ] Local execution completes without errors (with mock/test data)
4. [ ] Azure Blob Storage integration verified (upload/download test)
5. [ ] Azure AI Search integration verified (index creation, document upload)
6. [ ] Azure OpenAI classification returns valid scores (0-1 range)
7. [ ] Slack message posts successfully to configured channel
8. [ ] Bicep templates validate without errors
9. [ ] No console errors during execution
10. [ ] Existing project tests still pass
11. [ ] Documentation updated in README or dedicated docs

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests
| Test | File | What to Verify |
|------|------|----------------|
| Config Loading | `src/rfp_radar/tests/test_config.py` | All env vars load correctly, defaults work |
| Classifier | `src/rfp_radar/tests/test_classifier.py` | Relevance scores in 0-1 range, tags extracted |
| Proposal Generator | `src/rfp_radar/tests/test_proposal_generator.py` | Markdown output valid, branding included |
| Digest Builder | `src/rfp_radar/tests/test_digest_builder.py` | Slack message format valid |
| Models | `src/rfp_radar/tests/test_models.py` | Pydantic validation works |
| Base Scraper | `src/rfp_radar/tests/test_scrapers.py` | Interface contract enforced |

### Integration Tests
| Test | Services | What to Verify |
|------|----------|----------------|
| Storage Upload | rfp-radar ↔ Azure Blob | PDF upload, metadata storage |
| Search Index | rfp-radar ↔ Azure AI Search | Document indexing, search query |
| LLM Classification | rfp-radar ↔ Azure OpenAI | API call success, response parsing |
| Slack Delivery | rfp-radar ↔ Slack API | Message posted, formatting correct |

### End-to-End Tests
| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| Full Pipeline | 1. Run main.py with test data 2. Check Blob Storage 3. Check Search Index 4. Check Slack | RFP processed, proposal generated, digest sent |
| Empty Results | 1. Run with no matching RFPs 2. Check Slack | "No new RFPs" message sent |
| Error Handling | 1. Run with invalid credentials 2. Check logs | Graceful failure with error logs |

### Browser Verification (if frontend)
N/A - This is a batch processing service without frontend.

### Database Verification (if applicable)
| Check | Query/Command | Expected |
|-------|---------------|----------|
| Blob Container Exists | `az storage container exists --name rfp-radar` | `{"exists": true}` |
| Search Index Exists | `az search index show --name rfp-radar-index` | Index definition returned |
| Blob Documents | `az storage blob list --container-name rfp-radar` | RFP documents listed |

### QA Sign-off Requirements
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All E2E tests pass
- [ ] Local execution verified with test data
- [ ] Docker container builds and runs
- [ ] Azure resources created via Bicep
- [ ] Slack integration verified
- [ ] No regressions in existing functionality
- [ ] Code follows established patterns (Flake8 compliant)
- [ ] No security vulnerabilities introduced (no hardcoded secrets)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     NAITIVE RFP Radar                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  GovTribe    │  │   OpenGov    │  │    BidNet    │          │
│  │  Scraper     │  │   Scraper    │  │   Scraper    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────────────┬┴─────────────────┘                  │
│                          ▼                                      │
│                  ┌───────────────┐                              │
│                  │    Filter     │ (Age ≤ 3 days, US only)     │
│                  └───────┬───────┘                              │
│                          ▼                                      │
│                  ┌───────────────┐      ┌──────────────────┐   │
│                  │  Classifier   │◄────►│  Azure OpenAI    │   │
│                  │ (Score + Tags)│      │  (GPT-4o)        │   │
│                  └───────┬───────┘      └──────────────────┘   │
│                          ▼                                      │
│                  ┌───────────────┐                              │
│                  │    Filter     │ (Score ≥ 0.55)              │
│                  └───────┬───────┘                              │
│                          ▼                                      │
│         ┌────────────────┼────────────────┐                    │
│         ▼                ▼                ▼                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │ Blob Storage │ │  AI Search   │ │   Proposal   │           │
│  │ (PDFs/Meta)  │ │  (Index)     │ │  Generator   │           │
│  └──────────────┘ └──────────────┘ └──────┬───────┘           │
│                                           ▼                    │
│                                   ┌──────────────┐             │
│                                   │    Slack     │             │
│                                   │   #bots      │             │
│                                   └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Appendix: File Structure

```
src/rfp_radar/
├── __init__.py
├── main.py                 # Main orchestrator
├── config.py               # Environment configuration
├── models.py               # Pydantic data models
├── classifier.py           # AI relevance scoring
├── proposal_generator.py   # Level 3 proposal creation
├── digest_builder.py       # Slack message formatting
├── storage_client.py       # Azure Blob client
├── search_client.py        # Azure AI Search client
├── slack_client.py         # Slack SDK wrapper
├── llm_client.py           # Azure OpenAI REST client
├── logging_utils.py        # Structured logging
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── scrapers/
│   ├── __init__.py         # Scraper registry
│   ├── base.py             # Base scraper class
│   ├── govtribe.py         # GovTribe scraper
│   ├── opengov.py          # OpenGov scraper
│   └── bidnet.py           # BidNet scraper
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_classifier.py
    ├── test_proposal_generator.py
    ├── test_digest_builder.py
    ├── test_models.py
    └── test_scrapers.py

infra/
└── modules/
    └── rfp-radar-job.bicep  # Container Apps Job definition

.github/workflows/
└── rfp-radar-deploy.yml     # CI/CD workflow
```

## Appendix: Dependencies

```txt
# src/rfp_radar/requirements.txt
azure-storage-blob==12.23.0
azure-search-documents==11.6.0
requests>=2.32.3  # 2.32.4+ recommended for CVE-2024-47081 fix
python-dotenv==1.0.1
slack-sdk==3.33.1  # Note: install as 'slack-sdk' (hyphen), import as 'slack_sdk' (underscore)
pydantic==2.9.2
tqdm==4.66.5
```
