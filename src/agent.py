import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
import httpx
from livekit import rtc, api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero, simli
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


def _log_startup_config() -> None:
    def _presence(name: str) -> str:
        return "set" if os.getenv(name) else "missing"

    logger.info(
        "Startup env presence: LIVEKIT_URL=%s LIVEKIT_API_KEY=%s LIVEKIT_API_SECRET=%s "
        "BACKEND_URL=%s S3_ENDPOINT=%s S3_BUCKET=%s SIMLI_API_KEY=%s SIMLI_FACE_ID=%s",
        _presence("LIVEKIT_URL"),
        _presence("LIVEKIT_API_KEY"),
        _presence("LIVEKIT_API_SECRET"),
        _presence("BACKEND_URL"),
        _presence("S3_ENDPOINT"),
        _presence("S3_BUCKET"),
        _presence("SIMLI_API_KEY"),
        _presence("SIMLI_FACE_ID"),
    )


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""Вы — полезный голосовой ИИ-помощник. Пользователь общается с вами голосом.
            Вы охотно помогаете пользователям, отвечая на их вопросы, используя свои обширные знания.
            Ваши ответы должны быть краткими, по существу и без сложного форматирования, включая смайлики, звездочки или другие символы.
            Вы любознательны, дружелюбны и обладаете чувством юмора. Ответы должны быть на русском языке.""",
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm

async def send_to_server(data: dict):
    """Sends session data to your backend server."""
    backend_url = os.getenv("BACKEND_URL", "https://your-server.com/api/call-data")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(backend_url, json=data)
            if response.status_code != 200:
                logger.error(f"Server returned error {response.status_code}: {response.text}")
            response.raise_for_status()
            logger.info("Successfully sent call data to server")
    except Exception as e:
        logger.error(f"Failed to send call data to server: {e}")

async def start_egress(ctx: JobContext, audio_track_id: str, video_track_id: str):
    """Starts a LiveKit Egress job to record the person's audio and video and upload to Yandex."""
    lk_api = api.LiveKitAPI()
    try:
        room_name = ctx.room.name
        room_sid = await ctx.room.sid

        s3_config = api.S3Upload(
            endpoint=os.getenv("S3_ENDPOINT", "https://storage.yandexcloud.net"),
            bucket=os.getenv("S3_BUCKET"),
            access_key=os.getenv("S3_ACCESS_KEY"),
            secret=os.getenv("S3_SECRET"),
            region=os.getenv("S3_REGION", "ru-central1"),
            force_path_style=True # Important for Yandex/S3 compatibility
        )

        egress_request = api.TrackCompositeEgressRequest(
            room_name=room_name,
            audio_track_id=audio_track_id,
            video_track_id=video_track_id,
            file=api.EncodedFileOutput(
                filepath=f"recordings/{room_name}_{room_sid}.mp4",
                s3=s3_config
            )
        )
        await lk_api.egress.start_track_composite_egress(egress_request)
        logger.info(f"Started Egress to Yandex for room {ctx.room.name}")
    except Exception as e:
        logger.error(f"Failed to start Egress: {e}")
    finally:
        await lk_api.aclose()


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(
            model="deepgram/nova-3", 
            language="multi"
        ),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="qwen/qwen3-235b-a22b-instruct"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="inworld/inworld-tts-1-max", 
            voice="Dmitry", 
            language="ru"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns/
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    avatar = simli.AvatarSession(
      simli_config=simli.SimliConfig(
         api_key=os.getenv("SIMLI_API_KEY"),
         face_id=os.getenv("SIMLI_FACE_ID"),  # ID of the Simli face to use for your avatar. See "Face setup" for details.
      ),
   )

   # Start the avatar with retry logic (max 2 attempts)
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            await avatar.start(session, room=ctx.room)
            logger.info(f"Avatar started successfully on attempt {attempt + 1}")
            break
        except Exception as e:
            logger.warning(f"Avatar start attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in 2 seconds... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(2)
            else:
                logger.error("All avatar start attempts failed, continuing without avatar")
                # Continue with audio-only mode

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

    person_audio_track_id = None
    person_video_track_id = None
    egress_started = False

    async def maybe_start_egress():
        nonlocal person_audio_track_id, person_video_track_id, egress_started
        logger.info(f"Checking Egress: audio={person_audio_track_id}, video={person_video_track_id}, started={egress_started}")
        if person_audio_track_id and person_video_track_id and not egress_started:
            egress_started = True
            logger.info("Human tracks found, starting Egress...")
            asyncio.create_task(start_egress(ctx, person_audio_track_id, person_video_track_id))

    @ctx.room.on("track_published")
    def on_track_published(publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        nonlocal person_audio_track_id, person_video_track_id
        logger.info(f"Track published: kind={publication.kind}, participant={participant.identity}, participant_kind={participant.kind}")
        # Accept tracks from any participant that's not the agent itself
        if participant.identity != "simli-avatar-agent":
            if publication.kind == rtc.TrackKind.KIND_AUDIO:
                person_audio_track_id = publication.sid
                logger.info(f"Got audio track: {publication.sid}")
            elif publication.kind == rtc.TrackKind.KIND_VIDEO:
                person_video_track_id = publication.sid
                logger.info(f"Got video track: {publication.sid}")
            
            asyncio.create_task(maybe_start_egress())

    # Check for participants already in the room
    for participant in ctx.room.remote_participants.values():
        logger.info(f"Found participant: {participant.identity}, kind={participant.kind}")
        if participant.identity != "simli-avatar-agent":
            for publication in participant.track_publications.values():
                logger.info(f"Found existing track: {publication.kind}, sid={publication.sid}")
                if publication.kind == rtc.TrackKind.KIND_AUDIO:
                    person_audio_track_id = publication.sid
                elif publication.kind == rtc.TrackKind.KIND_VIDEO:
                    person_video_track_id = publication.sid
    
    await maybe_start_egress()

    async def finish_session():
        logger.info("Session finished, preparing report...")
        try:
            # Await async properties
            room_name = ctx.room.name
            room_sid = await ctx.room.sid
            
            report = ctx.make_session_report(session)
            data = {
                "room_name": room_name,
                "room_sid": room_sid,
                "transcript": report.to_dict(),
                "recording_path": f"recordings/{room_name}_{room_sid}.mp4"
            }
            await send_to_server(data)
        except Exception as e:
            logger.error(f"Error in finish_session: {e}")

    session.on("session_finished", lambda: asyncio.create_task(finish_session()))
    # Also add a shutdown callback as a safety net
    ctx.add_shutdown_callback(finish_session)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    _log_startup_config()
    try:
        cli.run_app(server)
    except Exception:
        logger.exception("Agent crashed during startup")
        raise
