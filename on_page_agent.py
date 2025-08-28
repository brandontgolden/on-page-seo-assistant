#!/usr/bin/env python3
"""

SEO Website Analyzer
A comprehensive Python-based on-page SEO analysis toolkit that provides automated website 
auditing and AI-powered content optimization recommendations.

"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from urllib.parse import urljoin, urlparse
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, cast, Any
import re
from datetime import datetime
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class RateLimiter:
    """
    Rate limiter with exponential backoff for handling API rate limits and 429 errors.
    """
    
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.request_count = 0
        self.last_request_time = 0
    
    def wait_for_rate_limit(self, min_interval: float = 0.5):
        """Ensure minimum time interval between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def exponential_backoff_with_jitter(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        delay = self.base_delay * (2 ** attempt)
        # Add jitter to avoid thundering herd
        jitter = random.uniform(0.1, 0.5) * delay
        return delay + jitter
    
    def should_retry(self, status_code: int, attempt: int) -> bool:
        """Determine if request should be retried based on status code and attempt count"""
        if attempt >= self.max_retries:
            return False
        
        # Retry on rate limiting, server errors, and timeouts
        retry_codes = {429, 500, 502, 503, 504, 408}
        return status_code in retry_codes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'seo_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SEOAnalysis:
    """Data class to hold SEO analysis results"""
    url: str
    page_title: str
    meta_description: str
    meta_keywords: str
    h1_tags: List[str]
    h2_tags: List[str]
    h3_tags: List[str]
    image_count: int
    images_without_alt: int
    internal_links: int
    external_links: int
    word_count: int
    title_problems: List[str]
    meta_problems: List[str]
    heading_problems: List[str]
    image_problems: List[str]
    content_problems: List[str]
    missing_elements: List[str]
    suggestions: List[str]
    canonical_url: str
    meta_robots: str
    schema_markup: bool
    page_load_time: float
    status_code: int
    error_message: str
    # AI-generated content
    ai_generated_title: str
    ai_generated_description: str
    ai_generated_keywords: str
    page_content_summary: str

class SEOWebsiteAnalyzer:
    """
    Comprehensive SEO analyzer that scans websites and provides detailed
    on-page SEO analysis with recommendations.
    """
    
    def __init__(self, base_url: str, batch_size: int = 100, max_workers: int = 10, openai_api_key: Optional[str] = None, 
                 request_delay: float = 0.5, max_retries: int = 5):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        logger.info(f"Initialized crawler for domain: {self.domain}")
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.analyzed_urls: Set[str] = set()
        self.product_urls: List[str] = []
        self.all_urls: List[str] = []
        
        # Initialize rate limiters with configurable settings
        self.http_rate_limiter = RateLimiter(max_retries=max_retries, base_delay=1.0)
        self.openai_rate_limiter = RateLimiter(max_retries=3, base_delay=2.0)
        
        logger.info(f"Rate limiting configured: {request_delay}s delay, {max_retries} max retries")
        
        # Initialize Azure OpenAI client
        self.openai_client = None
        
        # Try Azure OpenAI first, then fallback to standard OpenAI
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        
        if azure_api_key and azure_endpoint:
            try:
                self.openai_client = AzureOpenAI(
                    api_key=azure_api_key,
                    api_version=azure_api_version,
                    azure_endpoint=azure_endpoint
                )
                self.model_name = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o-mini')
                logger.info(f"Azure OpenAI client initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI: {e}")
                self.openai_client = None
        elif openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.model_name = os.getenv('OPENAI_MODEL', 'gpt-4')
                logger.info(f"Standard OpenAI client initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                self.openai_client = None
        else:
            logger.warning("No OpenAI API key provided. AI content generation will be skipped.")
            self.model_name = None
            
        # SEO expert system prompt
        self.seo_expert_prompt = """You are a SEO expert with 20 years of experience helping 
        ecommerce businesses grow through search optimization. You understand Google's algorithms, 
        user intent, and conversion optimization.

Your role is to analyze web page content and generate optimized SEO elements that will:
1. Improve search rankings
2. Increase click-through rates
3. Drive qualified traffic
4. Convert visitors to customers

Focus on ecommerce best practices, product optimization, and user experience."""
        
    def fetch_page(self, url: str, timeout: int = 10) -> tuple[Optional[BeautifulSoup], int, float, str]:
        """
        Fetch and parse a web page with robust retry logic and rate limiting.
        
        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (BeautifulSoup object, status_code, load_time, error_message)
        """
        start_time = time.time()
        last_status_code = 0
        last_error = ""
        
        for attempt in range(self.http_rate_limiter.max_retries + 1):
            try:
                # Apply rate limiting with configurable delay
                self.http_rate_limiter.wait_for_rate_limit(min_interval=self.request_delay)
                
                if attempt > 0:
                    delay = self.http_rate_limiter.exponential_backoff_with_jitter(attempt - 1)
                    logger.info(f"Retrying {url} (attempt {attempt + 1}/{self.http_rate_limiter.max_retries + 1}) after {delay:.2f}s delay")
                    time.sleep(delay)
                else:
                    logger.info(f"Fetching: {url}")
                
                response = self.session.get(url, timeout=timeout)
                last_status_code = response.status_code
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    load_time = time.time() - start_time
                    logger.info(f"Successfully fetched {url} in {load_time:.2f}s")
                    return soup, response.status_code, load_time, ""
                    
                elif response.status_code == 429:
                    # Rate limited - check for Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                            logger.warning(f"Rate limited (429) for {url}, waiting {wait_time}s as requested")
                            time.sleep(wait_time)
                        except ValueError:
                            pass
                    
                    last_error = f"Rate limited (HTTP 429) for {url}"
                    logger.warning(last_error)
                    
                    if not self.http_rate_limiter.should_retry(429, attempt):
                        break
                        
                elif self.http_rate_limiter.should_retry(response.status_code, attempt):
                    last_error = f"HTTP {response.status_code} error for {url}"
                    logger.warning(f"{last_error} - will retry")
                else:
                    # Non-retryable error
                    error_msg = f"HTTP {response.status_code} error for {url}"
                    logger.warning(error_msg)
                    return None, response.status_code, time.time() - start_time, error_msg
                    
            except requests.exceptions.Timeout:
                last_status_code = 408
                last_error = f"Timeout error for {url} after {timeout}s"
                if attempt < self.http_rate_limiter.max_retries:
                    logger.warning(f"{last_error} - will retry")
                else:
                    logger.error(last_error)
                    
            except requests.exceptions.ConnectionError:
                last_status_code = 0
                last_error = f"Connection error for {url}"
                if attempt < self.http_rate_limiter.max_retries:
                    logger.warning(f"{last_error} - will retry")
                else:
                    logger.error(last_error)
                    
            except Exception as e:
                last_status_code = 500
                last_error = f"Unexpected error for {url}: {str(e)}"
                if attempt < self.http_rate_limiter.max_retries:
                    logger.warning(f"{last_error} - will retry")
                else:
                    logger.error(last_error)
        
        # All retries exhausted
        final_error = f"Failed after {self.http_rate_limiter.max_retries + 1} attempts: {last_error}"
        logger.error(final_error)
        return None, last_status_code, time.time() - start_time, final_error

    def discover_urls(self, max_depth: int = 5, page_limit: Optional[int] = None) -> List[str]:
        """
        Discover all URLs on the website through crawling.
        
        Args:
            max_depth: Maximum crawl depth
            
        Returns:
            List of discovered URLs
        """
        urls_to_crawl = [self.base_url]
        discovered_urls = set([self.base_url])
        current_depth = 0
        
        while urls_to_crawl and current_depth < max_depth:
            # Check page limit
            if page_limit and len(discovered_urls) >= page_limit:
                logger.info(f"Reached page limit of {page_limit} URLs, stopping crawl")
                break
                
            logger.info(f"\n=== CRAWLING DEPTH {current_depth + 1}/{max_depth} ===")
            logger.info(f"Processing {len(urls_to_crawl)} URLs in this batch")
            logger.info(f"Total discovered so far: {len(discovered_urls)} URLs")
            current_batch = urls_to_crawl[:self.batch_size]
            urls_to_crawl = urls_to_crawl[self.batch_size:]
            new_urls = set()
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {executor.submit(self.fetch_page, url): url for url in current_batch}
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        if result[0]:  # soup exists
                            soup = result[0]
                            # Find all links with improved extraction
                            links = soup.find_all('a', href=True)
                            logger.debug(f"Found {len(links)} links on {url}")
                            
                            for link in links:
                                try:
                                    # More robust href extraction
                                    href = None
                                    if hasattr(link, 'get'):
                                        href = cast(Any, link).get('href')
                                    elif hasattr(link, 'href'):
                                        href = getattr(link, 'href', None)
                                    
                                    if not href or not isinstance(href, str):
                                        continue
                                        
                                    # Clean and normalize URL
                                    href = href.strip()
                                    if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                                        continue
                                        
                                    full_url = urljoin(url, href)
                                    
                                    # Remove URL fragments and normalize
                                    full_url = full_url.split('#')[0]
                                    parsed_url = urlparse(full_url)
                                    
                                    # Improved domain matching (handles www and subdomains)
                                    target_domain = self.domain.replace('www.', '')
                                    link_domain = parsed_url.netloc.replace('www.', '')
                                    
                                    # Skip non-HTML files
                                    excluded_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico', '.xml', '.txt', '.zip', '.doc', '.docx', '.xls', '.xlsx']
                                    if any(full_url.lower().endswith(ext) for ext in excluded_extensions):
                                        continue
                                    
                                    # Include same domain or relevant subdomains
                                    if (link_domain == target_domain or 
                                        link_domain.endswith('.' + target_domain)) and \
                                       full_url not in discovered_urls and \
                                       len(full_url) < 2000:  # Avoid extremely long URLs
                                        new_urls.add(full_url)
                                        discovered_urls.add(full_url)
                                        logger.debug(f"Added new URL: {full_url}")
                                except Exception as e:
                                    logger.debug(f"Error processing link {link}: {e}")
                                    continue
                    except Exception as e:
                        logger.error(f"Error processing {url}: {str(e)}")
            
            urls_to_crawl.extend(list(new_urls))
            current_depth += 1
            
            logger.info(f"Depth {current_depth}: Found {len(new_urls)} new URLs, {len(urls_to_crawl)} remaining to crawl, {len(discovered_urls)} total discovered")
            
            # Adaptive rate limiting between depth levels - be more respectful as we discover more URLs
            if current_depth < max_depth:
                depth_delay = min(1.0 + (len(discovered_urls) / 500), 5.0)  # Scale delay with URL count
                logger.info(f"Waiting {depth_delay:.2f}s before next crawl depth...")
                time.sleep(depth_delay)
        
        final_count = len(discovered_urls)
        logger.info(f"Crawling complete! Discovered {final_count} total URLs across {current_depth} depth levels")
        
        # Log a sample of discovered URLs for debugging
        sample_urls = list(discovered_urls)[:10]
        logger.info(f"Sample URLs discovered: {sample_urls}")
        
        return list(discovered_urls)

    def generate_seo_content_with_ai(self, url: str, page_content: str, existing_title: str = "", existing_description: str = "", existing_keywords: str = "") -> Dict[str, str]:
        """
        Generate optimized SEO content using OpenAI GPT.
        
        Args:
            url: Page URL
            page_content: Cleaned page content
            existing_title: Current title (if any)
            existing_description: Current meta description (if any)
            existing_keywords: Current meta keywords (if any)
            
        Returns:
            Dictionary with generated SEO content
        """
        if not self.openai_client:
            return {
                'title': 'AI generation not available (no API key)',
                'description': 'AI generation not available (no API key)',
                'keywords': 'AI generation not available (no API key)',
                'content_summary': 'AI generation not available (no API key)'
            }
        
        for attempt in range(self.openai_rate_limiter.max_retries + 1):
            try:
                # Apply rate limiting for OpenAI API
                self.openai_rate_limiter.wait_for_rate_limit(min_interval=1.0)
                
                if attempt > 0:
                    delay = self.openai_rate_limiter.exponential_backoff_with_jitter(attempt - 1)
                    logger.info(f"Retrying OpenAI API call for {url} (attempt {attempt + 1}) after {delay:.2f}s delay")
                    time.sleep(delay)
                
                # Clean and truncate content for API
                clean_content = re.sub(r'\s+', ' ', page_content[:3000])  # Limit to 3000 chars
                
                # Determine page type
                page_type = "product page" if any(indicator in url.lower() for indicator in ['/product/', '/products/', '/shop/', '/item/']) else "website page"
                
                prompt = f"""
{self.seo_expert_prompt}

Analyze this {page_type} and generate optimized SEO elements:

URL: {url}
Current Title: {existing_title or "MISSING"}
Current Meta Description: {existing_description or "MISSING"}
Current Keywords: {existing_keywords or "MISSING"}

Page Content:
{clean_content}

Generate the following optimized SEO elements:

1. TITLE: Create a compelling 50-60 character title that includes primary keywords and drives clicks
2. META_DESCRIPTION: Write a persuasive 150-160 character description that includes keywords and a call-to-action
3. KEYWORDS: List 8-12 relevant keywords/phrases (comma-separated) focusing on search intent and commercial value
4. CONTENT_SUMMARY: Provide a 2-sentence summary of what this page offers

Requirements:
- Focus on ecommerce conversion optimization
- Include location-based keywords if relevant
- Use action-oriented language
- Ensure keywords match user search intent
- Optimize for both SEO and human readability

Format your response as:
TITLE: [your title]
META_DESCRIPTION: [your description]
KEYWORDS: [keyword1, keyword2, keyword3, etc.]
CONTENT_SUMMARY: [your summary]
"""

                response = self.openai_client.chat.completions.create(
                    model=self.model_name or "gpt-4",
                    messages=[
                        {"role": "system", "content": self.seo_expert_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                # If we get here, the API call was successful
                break
                
            except Exception as api_error:
                error_str = str(api_error)
                
                # Check if it's a rate limit error
                if "rate" in error_str.lower() or "429" in error_str:
                    logger.warning(f"OpenAI API rate limited for {url}: {error_str}")
                    if attempt < self.openai_rate_limiter.max_retries:
                        continue
                    else:
                        logger.error(f"OpenAI API rate limit exceeded after {self.openai_rate_limiter.max_retries + 1} attempts")
                        return {
                            'title': 'AI generation failed: Rate limit exceeded',
                            'description': 'AI generation failed: Rate limit exceeded', 
                            'keywords': 'AI generation failed: Rate limit exceeded',
                            'content_summary': 'AI generation failed: Rate limit exceeded'
                        }
                elif "quota" in error_str.lower() or "billing" in error_str.lower():
                    logger.error(f"OpenAI API quota/billing issue: {error_str}")
                    return {
                        'title': 'AI generation failed: API quota exceeded',
                        'description': 'AI generation failed: API quota exceeded',
                        'keywords': 'AI generation failed: API quota exceeded', 
                        'content_summary': 'AI generation failed: API quota exceeded'
                    }
                else:
                    logger.warning(f"OpenAI API error for {url} (attempt {attempt + 1}): {error_str}")
                    if attempt < self.openai_rate_limiter.max_retries:
                        continue
                    else:
                        logger.error(f"OpenAI API failed after {self.openai_rate_limiter.max_retries + 1} attempts")
                        return {
                            'title': f'AI generation failed: {str(api_error)}',
                            'description': f'AI generation failed: {str(api_error)}',
                            'keywords': f'AI generation failed: {str(api_error)}',
                            'content_summary': f'AI generation failed: {str(api_error)}'
                        }
        
        # Parse the successful response
        try:
            ai_response = response.choices[0].message.content
            
            # Parse the response
            result = {
                'title': '',
                'description': '',
                'keywords': '',
                'content_summary': ''
            }
            
            lines = ai_response.split('\n') if ai_response else []
            for line in lines:
                line = line.strip()
                if line.startswith('TITLE:'):
                    result['title'] = line.replace('TITLE:', '').strip()
                elif line.startswith('META_DESCRIPTION:'):
                    result['description'] = line.replace('META_DESCRIPTION:', '').strip()
                elif line.startswith('KEYWORDS:'):
                    result['keywords'] = line.replace('KEYWORDS:', '').strip()
                elif line.startswith('CONTENT_SUMMARY:'):
                    result['content_summary'] = line.replace('CONTENT_SUMMARY:', '').strip()
            
            logger.info(f"Generated SEO content for {url}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing AI response for {url}: {str(e)}")
            return {
                'title': f'AI parsing failed: {str(e)}',
                'description': f'AI parsing failed: {str(e)}',
                'keywords': f'AI parsing failed: {str(e)}',
                'content_summary': f'AI parsing failed: {str(e)}'
            }
    
    def extract_clean_content(self, soup: BeautifulSoup) -> str:
        """
        Extract clean, readable content from the page for AI analysis.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Clean text content
        """
        if not soup:
            return ""
        
        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Extract main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))
        
        if main_content:
            text = main_content.get_text()
        else:
            text = soup.get_text()
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _identify_product_pages(self, urls: List[str]) -> List[str]:
        """
        Identify product pages from the list of URLs.
        
        Args:
            urls: List of URLs to analyze
            
        Returns:
            List of product page URLs
        """
        product_indicators = [
            '/product/', '/products/', '/shop/', '/item/', '/buy/',
            'product-', 'item-', '/p/', '/catalog/', '/store/'
        ]
        
        product_urls = []
        for url in urls:
            url_lower = url.lower()
            if any(indicator in url_lower for indicator in product_indicators):
                product_urls.append(url)
        
        logger.info(f"Identified {len(product_urls)} potential product pages")
        return product_urls

    def analyze_seo_elements(self, url: str, soup: BeautifulSoup, status_code: int, load_time: float, error_msg: str) -> SEOAnalysis:
        """
        Perform comprehensive SEO analysis on a single page.
        
        Args:
            url: Page URL
            soup: BeautifulSoup object of the page
            status_code: HTTP status code
            load_time: Page load time
            error_msg: Any error message
            
        Returns:
            SEOAnalysis object with all findings
        """
        if not soup:
            return SEOAnalysis(
                url=url, page_title="", meta_description="", meta_keywords="", h1_tags=[], h2_tags=[], h3_tags=[],
                image_count=0, images_without_alt=0, internal_links=0, external_links=0,
                word_count=0, title_problems=["Page failed to load"], meta_problems=[],
                heading_problems=[], image_problems=[], content_problems=[],
                missing_elements=[], suggestions=[], canonical_url="", meta_robots="",
                schema_markup=False, page_load_time=load_time, status_code=status_code,
                error_message=error_msg, ai_generated_title="", ai_generated_description="",
                ai_generated_keywords="", page_content_summary=""
            )

        # Extract basic elements
        title_tag = soup.find('title')
        page_title = title_tag.get_text().strip() if title_tag else ""
        
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_desc_tag:
            try:
                content = cast(Any, meta_desc_tag).get('content', '') if hasattr(meta_desc_tag, 'get') else getattr(meta_desc_tag, 'content', '')
                meta_description = content.strip() if isinstance(content, str) else str(content).strip() if content else ""
            except (AttributeError, TypeError):
                meta_description = ""
        else:
            meta_description = ""
        
        meta_keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords_tag:
            try:
                content = cast(Any, meta_keywords_tag).get('content', '') if hasattr(meta_keywords_tag, 'get') else getattr(meta_keywords_tag, 'content', '')
                meta_keywords = content.strip() if isinstance(content, str) else str(content).strip() if content else ""
            except (AttributeError, TypeError):
                meta_keywords = ""
        else:
            meta_keywords = ""
        
        # Extract headings
        h1_tags = [h.get_text().strip() for h in soup.find_all('h1')]
        h2_tags = [h.get_text().strip() for h in soup.find_all('h2')]
        h3_tags = [h.get_text().strip() for h in soup.find_all('h3')]
        
        # Analyze images
        images = soup.find_all('img')
        image_count = len(images)
        images_without_alt = 0
        for img in images:
            try:
                alt = cast(Any, img).get('alt', '') if hasattr(img, 'get') else getattr(img, 'alt', '')
                if not (alt.strip() if isinstance(alt, str) else str(alt).strip() if alt else ''):
                    images_without_alt += 1
            except (AttributeError, TypeError):
                images_without_alt += 1
        
        # Analyze links
        all_links = soup.find_all('a', href=True)
        internal_links = 0
        external_links = 0
        
        for link in all_links:
            try:
                href = cast(Any, link).get('href', '') if hasattr(link, 'get') else getattr(link, 'href', '')
                href_str = href if isinstance(href, str) else str(href) if href else ''
                if href_str:
                    if href_str.startswith('http') and self.domain not in href_str:
                        external_links += 1
                    elif self.domain in href_str or href_str.startswith('/') or not href_str.startswith('http'):
                        internal_links += 1
            except (AttributeError, TypeError):
                continue
        
        # Word count (approximate)
        text_content = soup.get_text()
        word_count = len(text_content.split())
        
        # Extract technical SEO elements
        canonical_tag = soup.find('link', rel='canonical')
        canonical_url = ""
        if canonical_tag:
            try:
                href = cast(Any, canonical_tag).get('href', '') if hasattr(canonical_tag, 'get') else getattr(canonical_tag, 'href', '')
                canonical_url = href if isinstance(href, str) else str(href) if href else ""
            except (AttributeError, TypeError):
                canonical_url = ""
        
        robots_tag = soup.find('meta', attrs={'name': 'robots'})
        meta_robots = ""
        if robots_tag:
            try:
                content = cast(Any, robots_tag).get('content', '') if hasattr(robots_tag, 'get') else getattr(robots_tag, 'content', '')
                meta_robots = content if isinstance(content, str) else str(content) if content else ""
            except (AttributeError, TypeError):
                meta_robots = ""
        
        # Check for schema markup
        schema_markup = bool(soup.find('script', type='application/ld+json') or 
                           soup.find(attrs={'itemscope': True}))
        
        # Generate AI content for missing elements
        page_content = self.extract_clean_content(soup)
        ai_content = {'title': '', 'description': '', 'keywords': '', 'content_summary': ''}
        
        # Only generate AI content if something is missing
        if not page_title or not meta_description or not meta_keywords:
            ai_content = self.generate_seo_content_with_ai(
                url=url,
                page_content=page_content,
                existing_title=page_title,
                existing_description=meta_description,
                existing_keywords=meta_keywords
            )
        # Analyze problems and suggestions
        title_problems = self.analyze_title_issues(page_title)
        meta_problems = self.analyze_meta_description_issues(meta_description)
        heading_problems = self.analyze_heading_issues(h1_tags, h2_tags, h3_tags)
        image_problems = self.analyze_image_issues(images, images_without_alt)
        content_problems = self.analyze_content_issues(word_count, text_content)
        missing_elements = self.identify_missing_elements(soup)
        suggestions = self.generate_suggestions(page_title, meta_description, h1_tags, word_count, schema_markup)
        
        return SEOAnalysis(
            url=url,
            page_title=page_title,
            meta_description=meta_description,
            meta_keywords=meta_keywords,
            h1_tags=h1_tags,
            h2_tags=h2_tags,
            h3_tags=h3_tags,
            image_count=image_count,
            images_without_alt=images_without_alt,
            internal_links=internal_links,
            external_links=external_links,
            word_count=word_count,
            title_problems=title_problems,
            meta_problems=meta_problems,
            heading_problems=heading_problems,
            image_problems=image_problems,
            content_problems=content_problems,
            missing_elements=missing_elements,
            suggestions=suggestions,
            canonical_url=canonical_url,
            meta_robots=meta_robots,
            schema_markup=schema_markup,
            page_load_time=load_time,
            status_code=status_code,
            error_message=error_msg,
            ai_generated_title=ai_content['title'],
            ai_generated_description=ai_content['description'],
            ai_generated_keywords=ai_content['keywords'],
            page_content_summary=ai_content['content_summary']
        )

    def analyze_title_issues(self, title: str) -> List[str]:
        """Analyze title tag issues"""
        problems = []
        if not title:
            problems.append("Missing title tag")
        elif len(title) < 30:
            problems.append("Title too short (under 30 characters)")
        elif len(title) > 60:
            problems.append("Title too long (over 60 characters)")
        
        if title and not any(char.isupper() for char in title):
            problems.append("Title lacks proper capitalization")
            
        return problems

    def analyze_meta_description_issues(self, meta_desc: str) -> List[str]:
        """Analyze meta description issues"""
        problems = []
        if not meta_desc:
            problems.append("Missing meta description")
        elif len(meta_desc) < 120:
            problems.append("Meta description too short (under 120 characters)")
        elif len(meta_desc) > 160:
            problems.append("Meta description too long (over 160 characters)")
            
        return problems

    def analyze_heading_issues(self, h1_tags: List[str], h2_tags: List[str], h3_tags: List[str]) -> List[str]:
        """Analyze heading structure issues"""
        problems = []
        if not h1_tags:
            problems.append("Missing H1 tag")
        elif len(h1_tags) > 1:
            problems.append("Multiple H1 tags found")
            
        if not h2_tags and h3_tags:
            problems.append("H3 tags present without H2 tags (poor heading hierarchy)")
            
        return problems

    def analyze_image_issues(self, images: List, images_without_alt: int) -> List[str]:
        """Analyze image-related issues"""
        problems = []
        if images_without_alt > 0:
            problems.append(f"{images_without_alt} images missing alt text")
        # Future enhancement: analyze image sizes, formats, etc.
        _ = images  # Mark as intentionally unused
        return problems

    def analyze_content_issues(self, word_count: int, text_content: str) -> List[str]:
        """Analyze content-related issues"""
        problems = []
        if word_count < 300:
            problems.append("Content too thin (under 300 words)")
        elif word_count > 3000:
            problems.append("Content very long (over 3000 words) - consider breaking up")
        # Future enhancement: analyze readability, keyword density, etc.
        _ = text_content  # Mark as intentionally unused
        return problems

    def identify_missing_elements(self, soup: BeautifulSoup) -> List[str]:
        """Identify missing SEO elements"""
        missing = []
        
        if not soup.find('meta', attrs={'name': 'description'}):
            missing.append("Meta description")
        if not soup.find('meta', attrs={'name': 'keywords'}):
            missing.append("Meta keywords")
        if not soup.find('link', rel='canonical'):
            missing.append("Canonical URL")
        if not soup.find('meta', property='og:title'):
            missing.append("Open Graph title")
        if not soup.find('meta', property='og:description'):
            missing.append("Open Graph description")
        if not soup.find('meta', property='og:image'):
            missing.append("Open Graph image")
        if not soup.find('script', type='application/ld+json'):
            missing.append("JSON-LD structured data")
            
        return missing

    def generate_suggestions(self, title: str, meta_desc: str, h1_tags: List[str], word_count: int, schema_markup: bool) -> List[str]:
        """Generate actionable SEO suggestions"""
        suggestions = []
        
        if not title or len(title) < 30:
            suggestions.append("Create compelling 30-60 character title with target keywords")
        if not meta_desc or len(meta_desc) < 120:
            suggestions.append("Write descriptive 120-160 character meta description")
        if not h1_tags:
            suggestions.append("Add H1 tag with primary keyword")
        if word_count < 300:
            suggestions.append("Expand content to at least 300 words")
        if not schema_markup:
            suggestions.append("Implement structured data markup for better search visibility")
            
        return suggestions

    def process_urls_in_batches(self, urls: List[str]) -> List[SEOAnalysis]:
        """
        Process URLs in parallel batches with rate limiting.
        
        Args:
            urls: List of URLs to analyze
            
        Returns:
            List of SEO analysis results
        """
        all_results = []
        total_batches = (len(urls) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(urls), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch = urls[i:i + self.batch_size]
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} URLs)")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {executor.submit(self.fetch_page, url): url for url in batch}
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        if result and len(result) == 4:
                            soup, status_code, load_time, error_msg = result
                            if soup is not None:
                                analysis = self.analyze_seo_elements(url, soup, status_code, load_time, error_msg)
                            else:
                                # Create error result for failed page fetch
                                analysis = SEOAnalysis(
                                    url=url, page_title="", meta_description="", meta_keywords="", h1_tags=[], h2_tags=[], h3_tags=[],
                                    image_count=0, images_without_alt=0, internal_links=0, external_links=0,
                                    word_count=0, title_problems=["Failed to fetch page"], meta_problems=[],
                                    heading_problems=[], image_problems=[], content_problems=[],
                                    missing_elements=[], suggestions=[], canonical_url="", meta_robots="",
                                    schema_markup=False, page_load_time=load_time, status_code=status_code,
                                    error_message=error_msg, ai_generated_title="", ai_generated_description="",
                                    ai_generated_keywords="", page_content_summary=""
                                )
                        else:
                            continue
                        all_results.append(analysis)
                        logger.info(f"Analyzed: {url}")
                    except Exception as e:
                        logger.error(f"Error analyzing {url}: {str(e)}")
                        # Create error result
                        error_analysis = SEOAnalysis(
                            url=url, page_title="", meta_description="", meta_keywords="", h1_tags=[], h2_tags=[], h3_tags=[],
                            image_count=0, images_without_alt=0, internal_links=0, external_links=0,
                            word_count=0, title_problems=[f"Analysis failed: {str(e)}"], meta_problems=[],
                            heading_problems=[], image_problems=[], content_problems=[],
                            missing_elements=[], suggestions=[], canonical_url="", meta_robots="",
                            schema_markup=False, page_load_time=0, status_code=0,
                            error_message=str(e), ai_generated_title="", ai_generated_description="",
                            ai_generated_keywords="", page_content_summary=""
                        )
                        all_results.append(error_analysis)
            
            # Adaptive rate limiting between batches
            if i + self.batch_size < len(urls):
                # Increase delay based on total request count to be more respectful
                base_delay = 2.0
                adaptive_delay = min(base_delay + (self.http_rate_limiter.request_count / 1000), 10.0)
                logger.info(f"Waiting {adaptive_delay:.2f} seconds between batches... (processed {self.http_rate_limiter.request_count} requests)")
                time.sleep(adaptive_delay)
        
        return all_results

    def export_to_csv(self, results: List[SEOAnalysis], filename: Optional[str] = None) -> str:
        """
        Export analysis results to CSV.
        
        Args:
            results: List of SEO analysis results
            filename: Optional custom filename
            
        Returns:
            Path to the created CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"seo_analysis_{self.domain}_{timestamp}.csv"
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            row = {
                'URL': result.url,
                'Status Code': result.status_code,
                'Load Time (s)': round(result.page_load_time, 2),
                
                # Current SEO Elements
                'Current Page Title': result.page_title,
                'Title Status': 'MISSING' if not result.page_title else ('NEEDS WORK' if result.title_problems else 'OK'),
                'Title Length': len(result.page_title),
                'Current Meta Description': result.meta_description,
                'Meta Description Status': 'MISSING' if not result.meta_description else ('NEEDS WORK' if result.meta_problems else 'OK'),
                'Meta Description Length': len(result.meta_description),
                'Current Meta Keywords': result.meta_keywords,
                'Meta Keywords Status': 'MISSING' if not result.meta_keywords else 'PRESENT',
                
                # AI-Generated SEO Content
                'AI Generated Title': result.ai_generated_title,
                'AI Generated Meta Description': result.ai_generated_description,
                'AI Generated Keywords': result.ai_generated_keywords,
                'Page Content Summary': result.page_content_summary,
                
                # Page Structure
                'H1 Tags': ' | '.join(result.h1_tags),
                'H1 Count': len(result.h1_tags),
                'H2 Count': len(result.h2_tags),
                'H3 Count': len(result.h3_tags),
                'Word Count': result.word_count,
                
                # Images and Links
                'Image Count': result.image_count,
                'Images Without Alt': result.images_without_alt,
                'Internal Links': result.internal_links,
                'External Links': result.external_links,
                
                # Technical SEO
                'Canonical URL': result.canonical_url,
                'Meta Robots': result.meta_robots,
                'Schema Markup': result.schema_markup,
                
                # Problems and Suggestions
                'Title Problems': ' | '.join(result.title_problems),
                'Meta Problems': ' | '.join(result.meta_problems),
                'Heading Problems': ' | '.join(result.heading_problems),
                'Image Problems': ' | '.join(result.image_problems),
                'Content Problems': ' | '.join(result.content_problems),
                'Missing Elements': ' | '.join(result.missing_elements),
                'SEO Suggestions': ' | '.join(result.suggestions),
                'Error Message': result.error_message
            }
            csv_data.append(row)
        
        # Write to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        logger.info(f"Results exported to {filename}")
        return filename

    def run_full_analysis(self, max_depth: int = 5, page_limit: Optional[int] = None) -> str:
        """
        Run the complete SEO analysis workflow.
        
        Returns:
            Path to the generated CSV report
        """
        logger.info(f"Starting SEO analysis for {self.base_url}")
        
        # Step 1: Discover all URLs
        logger.info("Step 1: Discovering website URLs...")
        logger.info(f"Starting crawl from: {self.base_url}")
        all_urls = self.discover_urls(max_depth=max_depth, page_limit=page_limit)
        
        if not all_urls:
            logger.error("No URLs discovered! Check if the website is accessible and has internal links.")
            return ""
        
        # Step 2: Identify product pages
        logger.info("Step 2: Identifying product pages...")
        product_urls = self._identify_product_pages(all_urls)
        
        # Step 3: Analyze all pages (prioritize product pages)
        logger.info("Step 3: Analyzing pages for SEO...")
        urls_to_analyze = product_urls + [url for url in all_urls if url not in product_urls]
        
        # Apply page limit if specified
        if page_limit and len(urls_to_analyze) > page_limit:
            logger.warning(f"Found {len(urls_to_analyze)} URLs, limiting to {page_limit} as requested")
            urls_to_analyze = urls_to_analyze[:page_limit]
        
        logger.info(f"Analyzing {len(urls_to_analyze)} total URLs ({len(product_urls)} product pages prioritized)")
        results = self.process_urls_in_batches(urls_to_analyze)
        
        # Step 4: Export results
        logger.info("Step 4: Exporting results to CSV...")
        csv_file = self.export_to_csv(results)
        
        logger.info(f"Analysis complete! Found {len(results)} pages analyzed.")
        logger.info(f"Results saved to: {csv_file}")
        
        return csv_file

def get_website_url() -> str:
    """Get website URL from user input with validation"""
    while True:
        print("\n" + "="*60)
        print("SEO Website Analyzer")
        print("="*60)
        
        website_url = input("Enter the website URL to analyze (e.g., https://example.com): ").strip()
        
        if not website_url:
            print("Error: Please enter a valid URL.")
            continue
            
        # Add https:// if no protocol specified
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
            print(f"Added HTTPS protocol: {website_url}")
        
        # Basic URL validation
        try:
            from urllib.parse import urlparse
            parsed = urlparse(website_url)
            if not parsed.netloc:
                print("Error: Invalid URL format. Please include the domain (e.g., https://example.com)")
                continue
        except Exception:
            print("Error: Invalid URL format.")
            continue
            
        # Confirm with user
        print(f"\nYou entered: {website_url}")
        confirm = input("Is this correct? (y/n): ").strip().lower()
        
        if confirm in ['y', 'yes']:
            return website_url
        elif confirm in ['n', 'no']:
            continue
        else:
            print("Please enter 'y' for yes or 'n' for no.")

def get_crawl_settings():
    """Get crawling configuration from user"""
    print("\n=== Crawling Configuration ===")
    
    # Ask about depth
    while True:
        try:
            depth = input("Max crawl depth (1-10, default 10): ").strip()
            if not depth:
                depth = 10
            else:
                depth = int(depth)
            if 1 <= depth <= 10:
                break
            else:
                print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")
    
    # Ask about page limit
    while True:
        try:
            limit = input("Max pages to analyze (default 20000, 0 for no limit): ").strip()
            if not limit:
                limit = 20000
            else:
                limit = int(limit)
            if limit >= 0:
                break
            else:
                print("Please enter a positive number or 0")
        except ValueError:
            print("Please enter a valid number")
    
    # Ask about rate limiting
    while True:
        try:
            delay = input("Request delay in seconds (0.1-5.0, default 0.5): ").strip()
            if not delay:
                delay = 0.5
            else:
                delay = float(delay)
            if 0.1 <= delay <= 5.0:
                break
            else:
                print("Please enter a number between 0.1 and 5.0")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            retries = input("Max retries for failed requests (1-10, default 5): ").strip()
            if not retries:
                retries = 5
            else:
                retries = int(retries)
            if 1 <= retries <= 10:
                break
            else:
                print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")
    
    return depth, limit if limit > 0 else None, delay, retries

def main():
    """Main function to run the SEO analyzer"""
    # Get website URL from user input
    WEBSITE_URL = get_website_url()
    
    # Get crawling configuration
    MAX_DEPTH, PAGE_LIMIT, REQUEST_DELAY, MAX_RETRIES = get_crawl_settings()
    
    # Configuration
    BATCH_SIZE = 100
    MAX_WORKERS = 10  # Adjust based on your system and target website capacity
    
    print(f"\nStarting SEO analysis for: {WEBSITE_URL}")
    print("This may take several minutes depending on website size...\n")
    
    # Check for AI configuration
    azure_key = os.getenv('AZURE_OPENAI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if azure_key:
        print("Azure OpenAI configuration detected - AI content generation enabled")
    elif openai_key:
        print("Standard OpenAI configuration detected - AI content generation enabled") 
    else:
        print("Warning: No AI API keys found in environment variables.")
        print("AI content generation will be skipped.")
        print("Set AZURE_OPENAI_API_KEY or OPENAI_API_KEY in your .env file")
        print()
    
    print(f"\nConfiguration:")
    print(f"- Max crawl depth: {MAX_DEPTH}")
    print(f"- Page limit: {PAGE_LIMIT or 'No limit'}")
    print(f"- Request delay: {REQUEST_DELAY}s")
    print(f"- Max retries: {MAX_RETRIES}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Max workers: {MAX_WORKERS}")
    
    # Initialize analyzer with robust rate limiting
    analyzer = SEOWebsiteAnalyzer(
        base_url=WEBSITE_URL,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS,
        openai_api_key=openai_key,  # Still pass as fallback
        request_delay=REQUEST_DELAY,
        max_retries=MAX_RETRIES
    )
    
    try:
        # Run analysis
        csv_file = analyzer.run_full_analysis(max_depth=MAX_DEPTH, page_limit=PAGE_LIMIT)
        print(f"\nSEO Analysis Complete!")
        print(f"Results saved to: {csv_file}")
        print(f"Detailed logs saved to: seo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        if azure_key or openai_key:
            print(f"AI-generated SEO content included for missing elements")
        else:
            print(f"AI content generation skipped (no API key provided)")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()