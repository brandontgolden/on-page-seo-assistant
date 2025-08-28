# On-Page SEO Agent

A comprehensive Python-based on-page SEO analysis toolkit that crawls websites, analyzes content, and generates AI-powered SEO optimization recommendations.

## Features

- **Intelligent Web Crawling**: Discovers all pages on a website through multi-depth crawling
- **Product Page Detection**: Automatically identifies and prioritizes e-commerce product pages
- **Comprehensive SEO Analysis**: Analyzes 25+ on-page SEO factors per page
- **AI Content Generation**: Uses OpenAI GPT-4 to create optimized titles, meta descriptions, and keywords
- **Rate Limiting & Retry Logic**: Respectful crawling with exponential backoff and configurable delays
- **Batch Processing**: Handles large websites efficiently with concurrent processing
- **Detailed Reporting**: Exports comprehensive CSV reports with actionable insights

## What It Analyzes

### Content Elements
- Page titles (length, optimization, capitalization)
- Meta descriptions (length, presence, quality)
- Meta keywords and content structure
- Heading hierarchy (H1, H2, H3 tags)
- Word count and content quality

### Technical Elements
- Images and alt text coverage
- Internal and external link analysis
- Canonical URLs and meta robots tags
- Schema markup detection
- Page load times and status codes

### AI-Generated Optimizations
- Optimized page titles (50-60 characters)
- Compelling meta descriptions (150-160 characters)
- Relevant keyword suggestions
- Page content summaries

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd seo_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with your OpenAI API credentials:

```env
# Azure OpenAI Configuration (recommended)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_MODEL=gpt-4o-mini

# Standard OpenAI Configuration (fallback)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
```

## Usage

Run the interactive SEO analyzer:

```bash
python on_page_agent.py
```

The script will prompt you to:
1. Enter the website URL to analyze
2. Configure crawling settings:
   - Maximum crawl depth (1-10)
   - Page limit (default 20,000)
   - Request delay between requests (0.1-5.0 seconds)
   - Maximum retry attempts (1-10)

### Example Configuration
```
Website URL: https://example.com
Max crawl depth: 3
Max pages: 500
Request delay: 0.5 seconds  
Max retries: 5
```

## Output Files

The analyzer generates several output files:

### CSV Report
`seo_analysis_{domain}_{timestamp}.csv` contains:
- **Current SEO Status**: URL, status codes, load times, existing titles/descriptions
- **AI-Generated Content**: Optimized titles, descriptions, keywords, content summaries
- **Technical Analysis**: Heading structure, image analysis, link counts
- **Issues & Recommendations**: Specific problems and actionable fix suggestions

### Log File
`seo_analysis_{timestamp}.log` contains detailed execution logs with timestamps.

## Report Structure

### Key CSV Columns

**Page Information:**
- URL, Status Code, Load Time
- Current Page Title, Meta Description, Keywords
- Title/Meta Status (OK, NEEDS WORK, MISSING)

**AI Optimizations:**
- AI Generated Title
- AI Generated Meta Description  
- AI Generated Keywords
- Page Content Summary

**Content Analysis:**
- H1/H2/H3 Tag counts and content
- Word Count, Image Count
- Images Without Alt Text
- Internal/External Link counts

**Technical SEO:**
- Canonical URL, Meta Robots
- Schema Markup presence
- Missing Elements (Open Graph, JSON-LD, etc.)

**Recommendations:**
- Title Problems, Meta Problems
- Heading Problems, Image Problems
- Content Problems, SEO Suggestions

## Architecture

### Core Components

1. **SEOWebsiteAnalyzer**: Main orchestrator class
2. **RateLimiter**: Handles respectful crawling with exponential backoff
3. **SEOAnalysis**: Data structure for analysis results
4. **AI Content Generator**: GPT-4 powered content optimization

### Key Features

- **Multi-threaded crawling** with configurable concurrency
- **Intelligent rate limiting** that adapts based on response
- **Robust error handling** with retry logic
- **Product page prioritization** for e-commerce sites
- **Comprehensive logging** for debugging and monitoring

## Performance & Limits

- **Default Settings**: 500ms delay, 10 concurrent workers, 5 retries
- **Page Discovery**: Up to 20,000 pages (configurable)
- **Crawl Depth**: Up to 10 levels (configurable)
- **Rate Limiting**: Adaptive delays based on website response
- **Memory Usage**: Processes pages in configurable batches

## AI Content Generation

The tool uses OpenAI GPT-4 to generate:
- SEO-optimized page titles with target keywords
- Compelling meta descriptions with calls-to-action
- Relevant keyword lists focused on search intent
- Content summaries for pages

AI generation only triggers for pages missing SEO elements, making it cost-efficient.

## Best Practices

1. **Start Small**: Test with small websites first
2. **Check robots.txt**: Ensure you're allowed to crawl
3. **Monitor Resources**: Watch CPU/memory during large crawls
4. **Adjust Rate Limiting**: Increase delays for slower websites
5. **API Usage**: Monitor OpenAI API costs and usage

## Error Handling

- Graceful handling of network timeouts and HTTP errors
- Comprehensive retry logic with exponential backoff
- Individual page failures don't stop batch processing
- Detailed error reporting in CSV output
- Rate limit detection and automatic backing off

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.