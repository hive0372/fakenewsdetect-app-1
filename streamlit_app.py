import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import re
from difflib import SequenceMatcher
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure page
st.set_page_config(
    page_title="Enhanced Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Enhanced trusted news sources with multiple endpoints
TRUSTED_SOURCES = {
    'NDTV': [
        'https://www.ndtv.com/latest',
        'https://www.ndtv.com/india-news',
        'https://www.ndtv.com/world-news'
    ],
    'India Today': [
        'https://www.indiatoday.in/latest-news',
        'https://www.indiatoday.in/india',
        'https://www.indiatoday.in/world'
    ],
    'Times of India': [
        'https://timesofindia.indiatimes.com/home/headlines',
        'https://timesofindia.indiatimes.com/india/news',
        'https://timesofindia.indiatimes.com/world/news'
    ],
    'The Hindu': [
        'https://www.thehindu.com/news/national/',
        'https://www.thehindu.com/news/international/'
    ],
    'Hindustan Times': [
        'https://www.hindustantimes.com/latest-news',
        'https://www.hindustantimes.com/india-news',
        'https://www.hindustantimes.com/world-news'
    ],
    'News18': [
        'https://www.news18.com/news/india/',
        'https://www.news18.com/news/world/'
    ]
}

class EnhancedNewsVerifier:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'])
    
    def extract_keywords(self, text):
        """Extract meaningful keywords from text"""
        # Convert to lowercase and remove punctuation
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stop words and short words
        keywords = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return keywords
    
    def keyword_similarity(self, text1, text2):
        """Calculate similarity based on common keywords"""
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0
    
    def clean_headline(self, headline):
        """Clean and normalize headline for comparison"""
        if not headline:
            return ""
        
        # Remove extra whitespace and convert to lowercase
        headline = re.sub(r'\s+', ' ', headline.strip().lower())
        
        # Remove common news prefixes and suffixes
        prefixes_to_remove = [
            'breaking:', 'live:', 'watch:', 'read:', 'exclusive:', 'update:',
            'news:', 'latest:', 'trending:', 'viral:', 'photos:', 'video:'
        ]
        
        for prefix in prefixes_to_remove:
            if headline.startswith(prefix):
                headline = headline[len(prefix):].strip()
        
        return headline
    
    def enhanced_similarity(self, text1, text2):
        """Enhanced similarity calculation using multiple methods"""
        # Method 1: Direct string similarity
        direct_sim = SequenceMatcher(None, text1, text2).ratio()
        
        # Method 2: Keyword-based similarity
        keyword_sim = self.keyword_similarity(text1, text2)
        
        # Method 3: Word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        word_overlap = len(words1.intersection(words2)) / max(len(words1), len(words2), 1)
        
        # Weighted combination
        final_score = (direct_sim * 0.4) + (keyword_sim * 0.4) + (word_overlap * 0.2)
        
        return final_score
    
    def scrape_website(self, url, source_name):
        """Generic website scraper with improved selectors"""
        headlines = []
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Common headline selectors for different sites
            selectors = [
                # Generic selectors
                'h1', 'h2', 'h3',
                # Common class patterns
                '[class*="headline"]', '[class*="title"]', '[class*="story"]',
                '[class*="news"]', '[class*="article"]',
                # Link text
                'a[href*="/news/"]', 'a[href*="/story/"]', 'a[href*="/article/"]',
                # Specific patterns
                '.story-title', '.news-title', '.headline', '.entry-title'
            ]
            
            for selector in selectors:
                try:
                    elements = soup.select(selector)
                    for element in elements:
                        text = element.get_text(strip=True)
                        if self.is_valid_headline(text):
                            cleaned = self.clean_headline(text)
                            if cleaned and len(cleaned) > 10:
                                headlines.append(cleaned)
                                
                    if len(headlines) > 15:  # Stop if we have enough headlines
                        break
                        
                except Exception:
                    continue
            
            # Remove duplicates while preserving order
            seen = set()
            unique_headlines = []
            for headline in headlines:
                if headline not in seen:
                    seen.add(headline)
                    unique_headlines.append(headline)
                    
            return unique_headlines[:20]  # Return top 20
            
        except Exception as e:
            st.warning(f"Error scraping {source_name}: {str(e)}")
            return []
    
    def is_valid_headline(self, text):
        """Check if text looks like a valid news headline"""
        if not text or len(text.strip()) < 10:
            return False
            
        # Filter out navigation elements, ads, etc.
        invalid_patterns = [
            r'^(home|about|contact|privacy|terms)', 
            r'^(login|register|subscribe)',
            r'^(search|menu|navigation)',
            r'^\d+$',  # Just numbers
            r'^[^a-zA-Z]*$',  # No letters
            r'(advertisement|sponsored|promoted)',
            r'(cookies|gdpr|privacy policy)'
        ]
        
        text_lower = text.lower()
        for pattern in invalid_patterns:
            if re.search(pattern, text_lower):
                return False
                
        # Must contain some letters and reasonable length
        if len(re.findall(r'[a-zA-Z]', text)) < 5:
            return False
            
        return True
    
    def scrape_all_sources(self):
        """Scrape headlines from all trusted sources with improved error handling"""
        all_headlines = []
        successful_sources = []
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_source = {}
            
            for source_name, urls in TRUSTED_SOURCES.items():
                for url in urls:
                    future = executor.submit(self.scrape_website, url, source_name)
                    future_to_source[future] = (source_name, url)
            
            for future in as_completed(future_to_source, timeout=30):
                source_name, url = future_to_source[future]
                try:
                    headlines = future.result(timeout=10)
                    if headlines:
                        all_headlines.extend(headlines)
                        if source_name not in successful_sources:
                            successful_sources.append(source_name)
                except Exception as e:
                    st.warning(f"Failed to scrape {source_name}: {str(e)}")
        
        if successful_sources:
            st.success(f"✅ Successfully scraped from: {', '.join(successful_sources)}")
        else:
            st.error("❌ Failed to scrape from any source")
        
        # Remove duplicates
        unique_headlines = list(set(all_headlines))
        st.info(f"📊 Found {len(unique_headlines)} unique headlines from trusted sources")
        
        return unique_headlines
    
    def verify_headline(self, input_headline):
        """Enhanced headline verification with better matching"""
        if not input_headline or len(input_headline.strip()) < 5:
            return {
                'status': 'error',
                'message': 'Please provide a valid headline to verify.',
                'confidence': 0,
                'matches': []
            }
        
        cleaned_input = self.clean_headline(input_headline)
        
        # Get headlines from trusted sources
        with st.spinner("Fetching latest headlines from trusted sources..."):
            trusted_headlines = self.scrape_all_sources()
        
        if not trusted_headlines:
            return {
                'status': 'error',
                'message': 'Unable to fetch headlines from trusted sources. Please check your internet connection and try again.',
                'confidence': 0,
                'matches': []
            }
        
        # Find matches with enhanced similarity
        matches = []
        for headline in trusted_headlines:
            similarity = self.enhanced_similarity(cleaned_input, headline)
            if similarity > 0.15:  # Lower threshold to catch more potential matches
                matches.append({
                    'headline': headline,
                    'similarity': similarity,
                    'original_headline': headline
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Enhanced verdict logic
        if not matches:
            return {
                'status': 'not_verified',
                'message': 'No similar headlines found in trusted sources. This could be breaking news, region-specific news, or potentially fake news.',
                'confidence': 0,
                'matches': [],
                'recommendation': 'Cross-check with multiple news sources and verify the source credibility.'
            }
        
        best_match = matches[0]
        
        if best_match['similarity'] > 0.6:
            return {
                'status': 'verified',
                'message': 'Strong match found with trusted sources. This appears to be legitimate news.',
                'confidence': best_match['similarity'],
                'matches': matches[:5],
                'recommendation': 'High confidence - news appears legitimate.'
            }
        elif best_match['similarity'] > 0.35:
            return {
                'status': 'partially_verified',
                'message': 'Similar content found but not an exact match. The news topic appears legitimate but details may vary.',
                'confidence': best_match['similarity'],
                'matches': matches[:5],
                'recommendation': 'Medium confidence - verify specific details with original sources.'
            }
        else:
            return {
                'status': 'low_confidence',
                'message': 'Weak similarity with trusted sources. Exercise caution and verify independently.',
                'confidence': best_match['similarity'],
                'matches': matches[:5],
                'recommendation': 'Low confidence - requires additional verification.'
            }

# Initialize the verifier
@st.cache_resource
def get_verifier():
    return EnhancedNewsVerifier()

# Streamlit UI
def main():
    st.title("🔍 Fake News Detector")
    st.markdown("### Advanced verification against trusted Indian news sources")
    
    # Information about improvements
    with st.expander("🔄 Recent Improvements"):
        st.markdown("""
        **Detection Features:**
        - ✅ **Multiple similarity algorithms** (keyword matching, word overlap, string similarity)
        - ✅ **Improved web scraping** with better headline extraction
        - ✅ **Lower false positive rate** with refined thresholds
        - ✅ **Better headline cleaning** removes prefixes and noise
        - ✅ **Multiple endpoints** per news source for better coverage
        - ✅ **Enhanced error handling** with detailed feedback
        - ✅ **Keyword-based matching** for semantic similarity
        """)
    
    # Information about the tool
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        This enhanced tool uses multiple verification methods:
        
        **Trusted Sources:**
        - NDTV, India Today, Times of India, The Hindu, Hindustan Times, News18
        
        **Verification Methods:**
        1. **Direct String Matching** - Exact headline comparison
        2. **Keyword Analysis** - Semantic content matching  
        3. **Word Overlap** - Common terms analysis
        
        **Confidence Levels:**
        - 🟢 **High (60%+)**: Strong match with trusted sources
        - 🟡 **Medium (35-60%)**: Similar content, verify details  
        - 🔴 **Low (<35%)**: Weak similarity, requires verification
        """)
    
    # Input section
    st.markdown("---")
    headline_input = st.text_area(
        "Enter the headline you want to verify:",
        placeholder="Example: 'Prime Minister announces new policy initiative' or any news headline...",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        verify_button = st.button("🔍 Verify News", type="primary")
    
    with col2:
        if st.button("🧪 Test with Sample"):
            headline_input = "Prime Minister Modi announces new digital initiative for rural development"
            st.rerun()
    
    if verify_button and headline_input.strip():
        verifier = get_verifier()
        
        # Show input analysis
        st.markdown("---")
        st.subheader("📝 Input Analysis")
        cleaned_input = verifier.clean_headline(headline_input)
        keywords = verifier.extract_keywords(headline_input)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Headline:**")
            st.write(f"'{headline_input}'")
        with col2:
            st.write("**Key Terms Extracted:**")
            st.write(", ".join(keywords[:10]) if keywords else "No keywords found")
        
        # Verification process
        with st.spinner("🔍 Analyzing headline against trusted sources..."):
            result = verifier.verify_headline(headline_input)
        
        st.markdown("---")
        st.subheader("📊 Verification Results")
        
        # Display result based on status
        if result['status'] == 'verified':
            st.success(f"✅ **VERIFIED**: {result['message']}")
            confidence_color = "green"
            st.balloons()
        elif result['status'] == 'partially_verified':
            st.warning(f"⚠️ **PARTIALLY VERIFIED**: {result['message']}")
            confidence_color = "orange"
        elif result['status'] == 'low_confidence':
            st.info(f"🔍 **LOW CONFIDENCE**: {result['message']}")
            confidence_color = "blue"
        elif result['status'] == 'not_verified':
            st.error(f"❌ **NOT VERIFIED**: {result['message']}")
            confidence_color = "red"
        else:
            st.error(f"🔧 **ERROR**: {result['message']}")
            confidence_color = "gray"
        
        # Confidence score and recommendation
        if result['confidence'] > 0:
            st.markdown(f"**Confidence Score**: :{confidence_color}[{result['confidence']:.1%}]")
        
        if 'recommendation' in result:
            st.info(f"💡 **Recommendation**: {result['recommendation']}")
        
        # Show matches if any
        if result['matches']:
            st.subheader("🔗 Similar Headlines Found")
            
            matches_data = []
            for i, match in enumerate(result['matches'], 1):
                matches_data.append({
                    'Rank': i,
                    'Similarity': f"{match['similarity']:.1%}",
                    'Headline': match['headline'][:120] + "..." if len(match['headline']) > 120 else match['headline']
                })
            
            matches_df = pd.DataFrame(matches_data)
            st.dataframe(matches_df, use_container_width=True, hide_index=True)
            
            # Show top match details
            if result['matches'][0]['similarity'] > 0.3:
                st.subheader("🎯 Best Match Analysis")
                best_match = result['matches'][0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Your Headline:**")
                    st.write(f"'{headline_input}'")
                with col2:
                    st.write("**Best Match:**")
                    st.write(f"'{best_match['headline']}'")
        
        # Additional tips
        st.markdown("---")
        st.info("💡 **Pro Tips**: For breaking news or very recent events, wait a few hours and re-check. International news might not appear on Indian news sites immediately.")
        
    elif verify_button and not headline_input.strip():
        st.error("Please enter a headline to verify.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🛡️ Fake News Detector v2.0 | Always cross-verify important news with multiple sources"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
