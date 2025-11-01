import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import asyncio

# (aiohttp 임포트 제거)

# .env 파일 로드
load_dotenv()

# --- Discord 봇 설정 ---
intents = discord.Intents.default()
intents.message_content = True

# Bot 클래스 (웹 서버 관련 코드 제거)
class MyBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (web_runner 제거)

    async def setup_hook(self):
        """봇이 시작되기 직전에 실행되는 훅 (비동기)"""
        print("--- 봇 셋업 시작 ---")
        
        # 1. Cogs 로드
        cogs_dir = './fsk-News-Room/cogs'
        if not os.path.exists(cogs_dir):
            print(f"✗ '{cogs_dir}' 디렉토리를 찾을 수 없습니다. cog를 로드할 수 없습니다.")
            # cogs 디렉토리 생성
            try:
                os.makedirs(cogs_dir)
                print(f"✓ '{cogs_dir}' 디렉토리를 생성했습니다. News-Room.py 파일을 해당 디렉토리로 옮겨주세요.")
            except Exception as e:
                print(f"✗ '{cogs_dir}' 디렉토리 생성 실패: {e}")
            return
            
        for filename in os.listdir(cogs_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                try:
                    await self.load_extension(f'cogs.{filename[:-3]}')
                    print(f"✓ '{filename}' cog가 로드되었습니다.")
                except Exception as e:
                    print(f"✗ '{filename}' cog 로드 중 오류 발생: {e}")
        
        # 2. 슬래시 커맨드 동기화
        try:
            await self.tree.sync()
            print("✓ 슬래시 커맨드 동기화 완료!")
        except Exception as e:
            print(f"✗ 슬래시 커맨드 동기화 실패: {e}")


        # (aiohttp 웹 서버 설정 및 시작 부분 제거)

    # (close 메소드 제거)

bot = MyBot(command_prefix="!", intents=intents)

# (naver_callback 핸들러 제거)

@bot.event
async def on_ready():
    # setup_hook이 실행된 후 호출됨
    print(f"\n{bot.user.name} 봇이 성공적으로 로그인했습니다! (ID: {bot.user.id})")

# --- 봇 실행 ---
if __name__ == "__main__":
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    if bot_token:
        # bot.run()이 asyncio 이벤트 루프를 시작
        bot.run(bot_token)
    else:
        print("오류: .env 파일에서 DISCORD_BOT_TOKEN을 찾을 수 없습니다.")
