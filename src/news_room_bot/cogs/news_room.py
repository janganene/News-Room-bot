import os
import discord
from discord import app_commands
from discord.ext import commands, tasks
import google.generativeai as genai
import asyncio
import warnings
import datetime
import json
import aiohttp
import re
import html
import nltk
import logging
from logging.handlers import RotatingFileHandler
from newspaper import Article as Article3k
from playwright.async_api import async_playwright
from .models import init_db, NewsHistory
from sqlalchemy import select

# 로깅 설정
def setup_logger():
    """로거 설정 - 파일과 콘솔에 동시 출력"""
    logger = logging.getLogger('news_bot')
    logger.setLevel(logging.INFO)
    
    # 이미 핸들러가 있으면 추가하지 않음 (중복 방지)
    if logger.handlers:
        return logger
    
    # 로그 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 파일 핸들러 (최대 10MB, 5개 백업 파일 유지)
    file_handler = RotatingFileHandler(
        'news_bot.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 로거 초기화
logger = setup_logger()

nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.data.find('tokenizers/punkt_tab')

NEWSPAPER_AVAILABLE = True
PLAYWRIGHT_AVAILABLE = True

def get_news_provider(news_link: str) -> str:
    """뉴스 링크에서 언론사 이름 추출"""
    news_provider = None
    providers = {
        # 종합 일간지
        "chosun.com": "조선일보",
        "donga.com": "동아일보",
        "joongang.co.kr": "중앙일보",
        "joins.com": "중앙일보",
        "hani.co.kr": "한겨레",
        "kyunghyang.com": "경향신문",
        "khan.co.kr": "경향신문",
        "seoul.co.kr": "서울신문",
        "hankookilbo.com": "한국일보",
        "munhwa.com": "문화일보",
        "segye.com": "세계일보",
        "kmib.co.kr": "국민일보",
        "dt.co.kr": "디지털타임스",
        "naeil.com": "내일신문",
        
        # 통신사
        "yna.co.kr": "연합뉴스",
        "newsis.com": "뉴시스",
        "news1.kr": "뉴스1",
        "newsen.com": "뉴스엔",
        "moneytoday.co.kr": "머니투데이",
        
        # 인터넷 언론
        "pressian.com": "프레시안",
        "ohmynews.com": "오마이뉴스",
        "vop.co.kr": "민중의소리",
        "dailian.co.kr": "데일리안",
        "newdaily.co.kr": "뉴데일리",
        "mediatoday.co.kr": "미디어오늘",
        "sisain.co.kr": "시사IN",
        "wikitree.co.kr": "위키트리",
        "insight.co.kr": "인사이트",
        "newsof.co.kr": "뉴스오브",
        "newstapa.org": "뉴스타파",
        "newsnjoy.or.kr": "뉴스앤조이",
        "kukinews.com": "쿠키뉴스",
        "sportsq.co.kr": "스포츠Q",
        "breaknews.com": "브레이크뉴스",
        "dailysecu.com": "데일리시큐",
        "goodmorningcc.com": "굿모닝충청",
        "newsworker.co.kr": "뉴스워커",
        "newspower.co.kr": "뉴스파워",
        "newscj.com": "뉴스씨제이",
        "ablenews.co.kr": "에이블뉴스",
        "beminor.com": "비마이너",
        "rapportian.com": "라포르시안",
        "straightnews.co.kr": "스트레이트뉴스",
        
        # 경제지
        "mk.co.kr": "매일경제",
        "hankyung.com": "한국경제",
        "mt.co.kr": "머니투데이",
        "news.mt.co.kr": "머니투데이",
        "sedaily.com": "서울경제",
        "etoday.co.kr": "이투데이",
        "edaily.co.kr": "이데일리",
        "fnnews.com": "파이낸셜뉴스",
        "heraldcorp.com": "헤럴드경제",
        "ajunews.com": "아주경제",
        "newspim.com": "뉴스핌",
        "newsway.co.kr": "뉴스웨이",
        "thebell.co.kr": "더벨",
        "businesspost.co.kr": "비즈니스포스트",
        "chosunbiz.com": "조선비즈",
        "biz.chosun.com": "조선비즈",
        "news.einfomax.co.kr": "연합인포맥스",
        "infostock.co.kr": "인포스탁",
        "tfmedia.co.kr": "더팩트",
        "thefact.co.kr": "더팩트",
        "tfnews.co.kr": "TF뉴스",
        "businesskorea.co.kr": "비즈니스코리아",
        "greened.kr": "환경일보",
        "ebn.co.kr": "EBN",
        "fortunekorea.co.kr": "포춘코리아",
        "motorgraph.com": "모터그래프",
        "autotimes.co.kr": "오토타임즈",
        "autoview.co.kr": "오토뷰",
        "autotribune.co.kr": "오토트리뷴",
        "dailycar.co.kr": "데일리카",
        
        # IT/테크
        "zdnet.co.kr": "ZDNet코리아",
        "etnews.com": "전자신문",
        "ddaily.co.kr": "디지털데일리",
        "bloter.net": "블로터",
        "itdonga.com": "IT동아",
        "betanews.net": "베타뉴스",
        "aitimes.com": "AI타임스",
        "aitimes.kr": "AI타임스",
        "boannews.com": "보안뉴스",
        "itworld.co.kr": "ITWorld",
        "ciokorea.com": "CIO Korea",
        "techm.kr": "테크M",
        "epnc.co.kr": "전자부품",
        "thelec.kr": "더일렉",
        "thelec.net": "더일렉",
        "digitaltoday.co.kr": "디지털투데이",
        "aving.net": "아빙뉴스",
        "datanet.co.kr": "데이터넷",
        "comworld.co.kr": "컴퓨터월드",
        "webtoday.co.kr": "웹투데이",
        "itbiznews.com": "IT비즈뉴스",
        "iconews.co.kr": "아이콘뉴스",
        
        # 게임
        "inven.co.kr": "인벤",
        "thisisgame.com": "디스이즈게임",
        "ruliweb.com": "루리웹",
        "gameshot.net": "게임샷",
        "gamefocus.co.kr": "게임포커스",
        "gamemeca.com": "게임메카",
        "gametoc.co.kr": "게임톡",
        "dailyesports.com": "데일리e스포츠",
        "fomos.co.kr": "포모스",
        "gamechosun.co.kr": "게임조선",
        "khgames.co.kr": "경향게임스",
        
        # 방송사
        "kbs.co.kr": "KBS",
        "news.kbs.co.kr": "KBS뉴스",
        "mbc.co.kr": "MBC",
        "imnews.imbc.com": "MBC뉴스",
        "sbs.co.kr": "SBS",
        "news.sbs.co.kr": "SBS뉴스",
        "jtbc.co.kr": "JTBC",
        "news.jtbc.co.kr": "JTBC뉴스",
        "ytn.co.kr": "YTN",
        "mbn.co.kr": "MBN",
        "tvchosun.com": "TV조선",
        "ichannela.com": "채널A",
        "channela.com": "채널A",
        "news.chosun.com": "채널A",
        "ebs.co.kr": "EBS",
        "tbs.seoul.kr": "TBS",
        "obs.co.kr": "OBS",
        "wowtv.co.kr": "한국경제TV",
        "arirang.co.kr": "아리랑TV",
        "ntv.co.kr": "NTV",
        "gbs.or.kr": "경기방송",
        "gtb.co.kr": "경기티브이",
        "pbc.co.kr": "평화방송",
        "cpbc.co.kr": "평화방송",
        "bbs.or.kr": "불교방송",
        "febc.net": "극동방송",
        "gugakfm.co.kr": "국악방송",
        "tbsradio.co.kr": "교통방송",
        
        # 포털/뉴스 플랫폼
        "naver.com": "네이버",
        "news.naver.com": "네이버뉴스",
        "m.news.naver.com": "네이버뉴스",
        "daum.net": "다음",
        "news.v.daum.net": "다음뉴스",
        "news.daum.net": "다음뉴스",
        "media.daum.net": "다음뉴스",
        "nate.com": "네이트",
        "news.nate.com": "네이트뉴스",
        "zum.com": "줌",
        "news.zum.com": "줌뉴스",
        "news.google.com": "구글뉴스",
        "google.com/news": "구글뉴스",
        "msn.com": "MSN뉴스",
        "news.yahoo.co.kr": "야후뉴스",
        
        # 기타 뉴스 플랫폼
        "m-i.kr": "매일일보",
        "inthenews.co.kr": "인더뉴스",
    }
    
    for domain, name in providers.items():
        if domain in news_link:
            news_provider = name
            break
    
    if not news_provider:
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', news_link)
        if domain_match:
            domain = domain_match.group(1)
            news_provider = domain.split('.')[0].upper()
    
    return news_provider

def is_it_news(title: str, content: str) -> bool:
    """제목과 내용을 분석하여 IT 관련 뉴스인지 판단"""
    # 핵심 IT 키워드 (가중치: 3점)
    core_it_keywords = [
        '인공지능', 'AI', '머신러닝', '딥러닝', '챗GPT', 'ChatGPT', 'claude', '클로드',
        '소프트웨어', '프로그래밍', '코딩', '개발자', '앱개발',
        '사이버보안', '해킹', '랜섬웨어', '데이터유출',
        '블록체인', '암호화폐', '비트코인', '이더리움', 'NFT',
        '메타버스', 'VR', '가상현실', 'AR', '증강현실',
        '반도체', '칩', 'CPU', 'GPU', 'NPU',
        '클라우드', '데이터센터', 'SaaS', 'PaaS',
        '자율주행', '드론기술', '로봇공학',
        '오픈AI', 'OpenAI', '앤스로픽', 'Anthropic', '딥마인드',
        '빅데이터', '데이터분석', '알고리즘',
    ]
    
    # 일반 IT 키워드 (가중치: 2점)
    general_it_keywords = [
        '애플리케이션', '플랫폼', 'API',
        '5G', '6G', '통신기술', 'IoT',
        '게임개발', '게임엔진', 'e스포츠',
        '스마트폰', '태블릿', '웨어러블',
        '전기차', '배터리기술',
        '보안패치', '암호화', '인증',
        '스타트업', '테크기업', '유니콘',
        '디지털전환', 'DX', '디지털화',
    ]
    
    # 보조 IT 키워드 (가중치: 1점)
    support_it_keywords = [
        '구글', '애플', '마이크로소프트', '아마존', '메타', '테슬라',
        '네이버', '카카오', '삼성전자', 'SK하이닉스', 'LG전자',
        '기술', '서비스', '온라인', '인터넷', '웹', '디지털',
        'IT', '정보기술', '게임', 'PC',
        '데이터', '네트워크', '보안',
    ]
    
    # 제외할 키워드 (경제/주식/정치 중심 뉴스)
    exclude_keywords = [
        '주가', '시세', '상장', 'IPO', '코스피', '코스닥', '증시',
        '투자', '매수', '매도', '수익률', '배당', '주주총회',
        '분기실적', '영업이익', '순이익', '매출액', '실적발표',
        '증권', '펀드', '채권', '금리', '환율',
        '대통령', '국회', '정치', '선거', '의원',
        '부동산', '아파트', '집값',
    ]
    
    combined_text = f"{title} {content}".lower()
    
    # 제외 키워드가 2개 이상 있으면 IT 뉴스가 아님
    exclude_count = sum(1 for keyword in exclude_keywords if keyword.lower() in combined_text)
    if exclude_count >= 2:
        return False
    
    # 가중치 기반 점수 계산
    score = 0
    
    # 핵심 IT 키워드 체크 (3점)
    for keyword in core_it_keywords:
        if keyword.lower() in combined_text:
            score += 3
    
    # 일반 IT 키워드 체크 (2점)
    for keyword in general_it_keywords:
        if keyword.lower() in combined_text:
            score += 2
    
    # 보조 IT 키워드 체크 (1점)
    for keyword in support_it_keywords:
        if keyword.lower() in combined_text:
            score += 1
    
    # 제목에 핵심 키워드가 있으면 보너스 점수 (2점)
    title_lower = title.lower()
    for keyword in core_it_keywords:
        if keyword.lower() in title_lower:
            score += 2
            break
    
    # 점수가 5점 이상이면 IT 뉴스로 판단
    return score >= 5

async def extract_article_content(url: str) -> tuple[str, str]:
    """
    뉴스 기사 본문 추출 (3단계 폴백)
    1. newspaper4k 시도
    2. playwright 시도
    3. 실패 시 None 반환
    
    Returns:
        tuple[str, str]: (추출된 본문, 사용된 방법)
    """
    # 1단계: newspaper4k 시도
    if NEWSPAPER_AVAILABLE and Article3k is not None:
        try:
            logger.info(f"  -> [1단계] newspaper로 본문 추출 시도...")
            article = Article3k(url, language='ko')
            
            # 비동기 처리를 위해 timeout 설정
            await asyncio.wait_for(
                asyncio.to_thread(article.download),
                timeout=10.0
            )
            await asyncio.wait_for(
                asyncio.to_thread(article.parse),
                timeout=5.0
            )
            
            content = article.text.strip()
            if content and len(content) > 100:  # 최소 길이 확인
                logger.info(f"     ✓ newspaper 성공 (길이: {len(content)})")
                return content, "newspaper"
            else:
                logger.warning(f"     ✗ newspaper 추출 실패 (길이 부족: {len(content)})")
        except asyncio.TimeoutError:
            logger.warning(f"     ✗ newspaper 타임아웃")
        except Exception as e:
            logger.error(f"     ✗ newspaper 오류: {e}")
    else:
        logger.info(f"  -> [1단계] newspaper 미설치 - 건너뜀")
    
    # 2단계: playwright 시도
    if PLAYWRIGHT_AVAILABLE:
        try:
            logger.info(f"  -> [2단계] playwright로 본문 추출 시도...")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # 타임아웃 설정
                await page.goto(url, timeout=15000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)  # 페이지 로딩 대기
                
                # 다양한 선택자로 본문 추출 시도
                selectors = [
                    'article',
                    '.article_body',
                    '.article-body',
                    '#articleBodyContents',
                    '#articeBody',
                    '.news_end',
                    '.article_view',
                    '#newsContent',
                    '.article-content',
                    'div[itemprop="articleBody"]',
                    '#content',
                    '.content'
                ]
                
                content = None
                for selector in selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            text = await element.inner_text()
                            if text and len(text.strip()) > 100:
                                content = text.strip()
                                logger.info(f"     ✓ playwright 성공 - 선택자: {selector} (길이: {len(content)})")
                                break
                    except:
                        continue
                
                await browser.close()
                
                if content:
                    return content, "playwright"
                else:
                    logger.warning(f"     ✗ playwright 추출 실패 (적절한 본문을 찾지 못함)")
                    
        except Exception as e:
            logger.error(f"     ✗ playwright 오류: {e}")
    else:
        logger.info(f"  -> [2단계] playwright 미설치 - 건너뜀")
    
    # 3단계: 모두 실패
    logger.info(f"  -> [3단계] 모든 본문 추출 방법 실패 - API 요약본 사용")
    return None, "failed"

class NewsCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
        self.NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
        self.NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
        self.CHANNEL_ID_STR = os.getenv("DISCORD_CHANNEL_ID")
        self.CHANNEL_ID = None

        if self.CHANNEL_ID_STR:
            try:
                self.CHANNEL_ID = int(self.CHANNEL_ID_STR)
            except ValueError:
                logger.error(f"오류: DISCORD_CHANNEL_ID가 올바른 숫자 형식이 아닙니다. (값: {self.CHANNEL_ID_STR})")
                return
        
        # JSON 파일 관련 코드 제거
        # self.history_file = "news_history.json"
        # self.sent_urls_history = {}
        
        # DB 세션 초기화
        self.db_session_maker = None
        
        self.model = self._initialize_ai_model()
        
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self._init_database())

    def _initialize_ai_model(self):
        """환경 변수 GEMINI_API_KEY로 AI 모델 초기화"""
        if not self.GEMINI_API_KEY:
            print("경고: GEMINI_API_KEY가 설정되지 않았습니다. AI 요약 기능이 작동하지 않습니다.")
            return None
        try:
            genai.configure(api_key=self.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("✓ Gemini 2.5 Flash 모델 초기화 완료.")
            return model
        except Exception as e:
            logger.error(f"✗ Gemini 모델 초기화 오류: {e}")
            return None
        
    async def _init_database(self):
        """데이터베이스 초기화 및 세션 메이커 생성"""
        await self.bot.wait_until_ready()
        try:
            self.db_session_maker = await init_db("news_history.db")
            
            # 기존 레코드 수 확인
            async with self.db_session_maker() as session:
                result = await session.execute(select(NewsHistory))
                count = len(result.scalars().all())
                logger.info(f"✓ 데이터베이스 초기화 완료. (총 {count}개 레코드)")
        except Exception as e:
            logger.error(f"데이터베이스 초기화 오류: {e}")

    # URL 확인 메서드 (새로 추가)
    async def _is_url_sent(self, url: str) -> bool:
        """URL이 이미 전송되었는지 확인"""
        if not self.db_session_maker:
            return False
        
        async with self.db_session_maker() as session:
            result = await session.execute(
                select(NewsHistory).where(NewsHistory.url == url)
            )
            return result.scalar_one_or_none() is not None

    # URL 저장 메서드 (새로 추가)
    async def _save_sent_url(self, url: str, message_id: str = None):
        """전송된 URL을 데이터베이스에 저장"""
        if not self.db_session_maker:
            return
        
        async with self.db_session_maker() as session:
            news_record = NewsHistory(url=url, message_id=message_id)
            session.add(news_record)
            await session.commit()

    async def fetch_naver_news(self, query="IT 기술 인공지능 소프트웨어 -경제 -주식 -투자", display=10):
        """네이버 뉴스 API에서 IT 기술 뉴스 검색 (더 많은 결과 가져오기)"""
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": self.NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": self.NAVER_CLIENT_SECRET
        }
        params = {
            "query": query,
            "display": display,
            "sort": "date"
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("items", [])
                    else:
                        error_text = await response.text()
                        logger.error(f"네이버 API 오류 발생: Status {response.status}, Error: {error_text}")
                        return []
            except Exception as e:
                logger.error(f"네이버 API 요청 중 오류 발생: {e}")
                return []

    async def summarize_with_ai(self, title, content):
        """AI로 뉴스 요약 (본문 내용 기반). 요청된 포맷을 따름."""
        if not self.model:
            return "AI 모델이 초기화되지 않아 요약할 수 없습니다."
            
        if not title or not content:
            return None
        
        max_length = 10000
        if len(content) > max_length:
            content = content[:max_length]
            
        try:
            prompt = f"""뉴스 내용을 아래 형식에 맞게 한국어로 요약해 주세요.

**요약 규칙:**
- 뉴스의 핵심만 2~3개 문단으로 요약합니다.
- 각 문단은 '>' 기호로 시작합니다.
- 문단과 문단 사이는 반드시 빈 줄 하나('> ')를 넣습니다.
- 다른 설명, 분석, 서론, 결론, 메타 정보 등은 절대 포함하지 않습니다.

**출력 형식:**
## **뉴스제목**
> 첫 번째 요약 문단
> 
> 두 번째 요약 문단
> 
> 세 번째 요약 문단(필요 시)

---
제목: {title}
내용: {content}
---
"""
            response = await asyncio.wait_for(
                self.model.generate_content_async(prompt),
                timeout=60.0
            )
            
            summary = response.text.strip()
            
            unwanted_phrases = [
                "물론입니다", "IT 전문 뉴스 에디터", "기사 내용을",
                "핵심만 담아", "전문적으로 요약", "---",
                "다음은", "요약입니다", "요약본입니다"
            ]
            
            lines = summary.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line_stripped = line.strip()
                
                if any(phrase in line_stripped for phrase in unwanted_phrases):
                    continue
                
                cleaned_lines.append(line)
            
            if cleaned_lines:
                return '\n'.join(cleaned_lines)
            
            return summary
            
        except asyncio.TimeoutError:
            logger.error("AI 요약 요청 시간 초과 (TimeoutError)")
            return f"오류: AI 요약 요청 시간 초과. 원문은 {title}."
        except Exception as e:
            logger.error(f"AI 요약 처리 중 오류: {e}")
            return None

    async def cog_load(self):
        """Cog가 로드될 때 실행"""
        if self.CHANNEL_ID:
            self.send_news_loop.start()
            logger.info("✓ 뉴스 자동 전송 루프 시작.")
        else:
            logger.warning("경고: DISCORD_CHANNEL_ID가 없어 자동 전송 루프를 시작하지 않습니다.")

    async def cog_unload(self):
        """Cog가 언로드될 때 실행"""
        self.send_news_loop.stop()
        logger.info("뉴스 자동 전송 루프 중지.")

    @tasks.loop(minutes=30.0)
    async def send_news_loop(self):
        """120분마다 최신 뉴스를 확인하고 채널에 전송"""
        await self.bot.wait_until_ready()
        logger.info("\n--- 뉴스 자동 업데이트 시작 ---")
        try:
            await self.fetch_and_send_news()
        except Exception as e:
            logger.error(f"자동 업데이트 루프 중 치명적 오류 발생: {e}", exc_info=True)
            
    async def fetch_and_send_news(self):
        """새 IT 기사를 찾아 AI 요약 후 새 메시지로 전송 또는 기존 메시지 수정"""
        if not self.NAVER_CLIENT_ID or not self.NAVER_CLIENT_SECRET or not self.CHANNEL_ID:
            logger.error("환경 변수(NAVER ID/SECRET/CHANNEL_ID) 미설정. 업데이트 중단.")
            return

        news_items = await self.fetch_naver_news(query="IT 기술 소프트웨어 인공지능", display=10)
        if not news_items:
            logger.info("새로운 IT 뉴스를 찾지 못했습니다. 다음 루프 대기.")
            return

        for item in news_items:
            news_url = item.get("originallink", "") or item.get("link", "")
            news_link = item.get("link", "")
            
            news_provider = get_news_provider(news_url) 

            title_with_provider = item.get("title", "")
            news_title = re.sub('<[^<]+?>', '', title_with_provider)
            news_title = html.unescape(news_title)  # HTML 엔티티 디코딩

            news_description = item.get("description", "")
            news_description = html.unescape(news_description)  # HTML 엔티티 디코딩
            
            # 이미 전송된 뉴스인지 확인 (DB 조회)
            if await self._is_url_sent(news_url):
                logger.info(f"  -> 이미 전송된 기사. 건너뜀: {news_provider} - {news_title}")
                continue

            logger.info(f"→ 새 기사 발견: {news_provider} - {news_title[:60]}...")
            
            # 본문 추출 시도
            content_to_summarize, extraction_method = await extract_article_content(news_url)
            
            if not content_to_summarize:
                content_to_summarize = re.sub('<[^<]+?>', '', news_description)
                extraction_method = "naver_api"
                logger.info(f"  -> API description 사용 (길이: {len(content_to_summarize)})")
            else:
                logger.info(f"  -> {extraction_method} 사용 (길이: {len(content_to_summarize)})")

            # IT 뉴스 필터링 (새 뉴스만)
            if not is_it_news(news_title, content_to_summarize):
                logger.info(f"  ✗ IT 뉴스가 아님. 건너뜀.")
                continue

            summary = await self.summarize_with_ai(news_title, content_to_summarize)
            if not summary or summary.startswith("오류:"):
                logger.warning("✗ 요약 실패. 다음 기사로 넘어갑니다.")
                continue

            # 대안: 제목 중복 방지 로직 추가
            current_year = datetime.datetime.now().year

            # summary에 이미 제목이 포함되어 있는지 확인
            if summary.startswith("## **"):
                news_text = f"{summary}\n"
            else:
                news_text = f"## **{news_title}**\n{summary}\n"

            news_text += f"\n"
            news_text += f"-# Published by Free Server Korea.\n"
            news_text += f"-# Copyright © {current_year} [{news_provider}]({news_url}) . All right reserved."

            try:
                channel = self.bot.get_channel(self.CHANNEL_ID)
                if not channel:
                    logger.error(f"오류: 채널 ID {self.CHANNEL_ID}를 찾을 수 없습니다.")
                    return

                # 새 메시지 전송
                message = await channel.send(news_text)
                logger.info(f"✓ 새 메시지 전송 완료 (ID: {message.id})")
                
                # 공지 채널인 경우 자동 발행
                if isinstance(channel, discord.TextChannel) and channel.is_news():
                    try:
                        await message.publish()
                        logger.info(f"✓ 메시지 자동 발행 완료 (ID: {message.id})")
                    except discord.errors.Forbidden:
                        logger.warning("⚠ 메시지 발행 권한이 없습니다.")
                    except discord.errors.HTTPException as e:
                        logger.warning(f"⚠ 메시지 발행 실패: {e}")
                
                # 전송 성공 후 DB에 저장
                await self._save_sent_url(news_url, str(message.id))
                
                # 전체 레코드 수 확인
                async with self.db_session_maker() as session:
                    result = await session.execute(select(NewsHistory))
                    count = len(result.scalars().all())
                    logger.info(f"✓ 데이터베이스 저장 완료. (총 {count}개)")
                
                # 하나 처리 후 종료
                return
            
            except discord.errors.Forbidden:
                logger.warning(f"오류: 채널 권한 없음. (채널 ID: {self.CHANNEL_ID})")
                return
            except Exception as e:
                logger.error(f"메시지 처리 오류: {e}", exc_info=True)

        logger.info("처리할 새로운 IT 뉴스를 찾지 못했습니다.")

async def setup(bot):

    await bot.add_cog(NewsCog(bot))
