import copy
import json
import os
import logging
import uuid
import httpx
import asyncio
from quart import (
    Blueprint,
    Quart,
    jsonify,
    make_response,
    request,
    send_from_directory,
    render_template,
    current_app,
)

from openai import AsyncAzureOpenAI
from azure.identity.aio import (
    DefaultAzureCredential,
    get_bearer_token_provider
)
from backend.auth.auth_utils import get_authenticated_user_details
from backend.security.ms_defender_utils import get_msdefender_user_json
from backend.history.cosmosdbservice import CosmosConversationClient
from backend.settings import (
    app_settings,
    MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
)
from backend.utils import (
    format_as_ndjson,
    format_stream_response,
    format_non_streaming_response,
    convert_to_pf_format,
    format_pf_non_streaming_response,
    BlockedTextScrubber,
)

import time
import datetime
try:
    from backend.permit_agent import agent as permit_agent
except Exception:  # missing aiohttp etc. -> feature simply stays off
    permit_agent = None
    logging.exception("permit agent unavailable; permit questions fall back to RAG")
try:  # City calendar (events + meetings) from events.json; replaces the Granicus meetings feed
    from backend import events as events_feed
except Exception:
    events_feed = None
    logging.exception("events feed unavailable; calendar questions fall back to RAG")
try:  # code pipeline for website questions (burbank-code-v1); opt-in via CODE_PIPELINE_ENABLED
    from backend import website_pipeline
except Exception:
    website_pipeline = None
    logging.exception("website pipeline unavailable; website questions use on-your-data")
CODE_PIPELINE_ENABLED = bool(website_pipeline) and os.environ.get("CODE_PIPELINE_ENABLED", "0") != "0"

try:  # address-specific zoning / land-use answers (answered from the code index); opt-in
    from backend import zoning
except Exception:
    zoning = None
    logging.exception("zoning route unavailable")
ZONING_ROUTE_ENABLED = bool(zoning) and os.environ.get("ZONING_ROUTE_ENABLED", "0") != "0"

# Arrest / police daily-log deflect. The logs update daily and can't be re-pushed that often, and
# the model won't reliably stop enumerating/inventing per-day dates from a prompt rule, so answer
# DETERMINISTICALLY: the classifier only routes here, the reply is fixed text (no LLM generation,
# no retrieval), which is why it can never fabricate dates or log contents. Env-configurable.
ARREST_ROUTE_ENABLED = os.environ.get("ARREST_ROUTE_ENABLED", "0") != "0"
ARREST_LOG_URL = os.environ.get(
    "ARREST_LOG_URL", "https://www.burbankca.gov/web/police-department/daily-arrest-logs")
ARREST_ANSWER = (
    "The Police Department's daily arrest logs are updated every day and posted on the City's "
    "official page:\n\n" + ARREST_LOG_URL + "\n\nOpen that page and select the date you want to "
    "view or download its log (the page covers roughly the past 30 days). The logs are public "
    "record."
)

bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")

cosmos_db_ready = asyncio.Event()


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    # Application Insights: emit request + dependency telemetry so the Performance (p50/p95) and
    # Failures blades populate. Only activates when APPLICATIONINSIGHTS_CONNECTION_STRING is set
    # (Azure injects it via the bicep app setting); local runs without it are a no-op. The OTel
    # distro auto-instruments Flask/FastAPI but not Quart, so we wrap the ASGI app by hand to get
    # server spans. Runs once per gunicorn worker (no preload_app), where the exporter must live.
    _appinsights_conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if _appinsights_conn:
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor
            from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
            configure_azure_monitor(connection_string=_appinsights_conn)
            app.asgi_app = OpenTelemetryMiddleware(app.asgi_app)
            logging.info("Application Insights instrumentation enabled.")
        except Exception:
            # Telemetry must never take the app down; degrade to no tracing.
            logging.exception("App Insights instrumentation skipped")

    @app.before_serving
    async def init():
        try:
            app.cosmos_conversation_client = await init_cosmosdb_client()
            cosmos_db_ready.set()
        except Exception as e:
            logging.exception("Failed to initialize CosmosDB client")
            app.cosmos_conversation_client = None
            raise e
    
    return app


@bp.route("/")
async def index():
    return await render_template(
        "index.html",
        title=app_settings.ui.title,
        favicon=app_settings.ui.favicon
    )


@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")


@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory("static/assets", path)


# Debug settings
DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

USER_AGENT = "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"


# Frontend Settings via Environment Variables
frontend_settings = {
    "auth_enabled": app_settings.base_settings.auth_enabled,
    "feedback_enabled": (
        app_settings.chat_history and
        app_settings.chat_history.enable_feedback
    ),
    "ui": {
        "title": app_settings.ui.title,
        "logo": app_settings.ui.logo,
        "chat_logo": app_settings.ui.chat_logo or app_settings.ui.logo,
        "chat_title": app_settings.ui.chat_title,
        "chat_description": app_settings.ui.chat_description,
        "chat_subtitle": app_settings.ui.chat_subtitle,
        "show_share_button": app_settings.ui.show_share_button,
        "show_chat_history_button": app_settings.ui.show_chat_history_button,
        "chat_response_contactmessage": app_settings.ui.chat_response_contactmessage,
        "poweredby": app_settings.ui.poweredby,
        "poweredbycomp": app_settings.ui.poweredbycomp,
        "poweredbyurl": app_settings.ui.poweredbyurl,
        "headertitle": app_settings.ui.headertitle,
        "example_title": app_settings.ui.example_title,
        "example_option_1": app_settings.ui.example_option_1,
        "example_option_2":  app_settings.ui.example_option_2,
        "example_option_3":  app_settings.ui.example_option_3,
        "example_option_4":  app_settings.ui.example_option_4,
        "capabilities":  app_settings.ui.capabilities,
        "capabilities_1":  app_settings.ui.capabilities_1,
        "capabilities_2": app_settings.ui.capabilities_2,
        "capabilities_3": app_settings.ui.capabilities_3,
        "capabilities_4": app_settings.ui.capabilities_4,
        "capabilities_5": app_settings.ui.capabilities_5,
        "limitations": app_settings.ui.limitations,
        "limitations_1": app_settings.ui.limitations_1,
        "limitations_2": app_settings.ui.limitations_2,
        "limitations_3": app_settings.ui.limitations_3,
        "limitations_4": app_settings.ui.limitations_4,
        "chat_resp_logo": app_settings.ui.chat_resp_logo,
        "hand_wave_icon": app_settings.ui.hand_wave_icon,
        "show_permit_link": app_settings.ui.show_permit_link,
        "speaker_icon": app_settings.ui.speaker_icon
    },
    "sanitize_answer": app_settings.base_settings.sanitize_answer,
    "oyd_enabled": app_settings.base_settings.datasource_type,
}


# Enable Microsoft Defender for Cloud Integration
MS_DEFENDER_ENABLED = os.environ.get("MS_DEFENDER_ENABLED", "true").lower() == "true"


# Initialize Azure OpenAI Client
async def init_openai_client():
    azure_openai_client = None
    
    try:
        # API version check
        if (
            app_settings.azure_openai.preview_api_version
            < MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
        ):
            raise ValueError(
                f"The minimum supported Azure OpenAI preview API version is '{MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION}'"
            )

        # Endpoint
        if (
            not app_settings.azure_openai.endpoint and
            not app_settings.azure_openai.resource
        ):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_RESOURCE is required"
            )

        endpoint = (
            app_settings.azure_openai.endpoint
            if app_settings.azure_openai.endpoint
            else f"https://{app_settings.azure_openai.resource}.openai.azure.com/"
        )

        # Authentication
        aoai_api_key = app_settings.azure_openai.key
        ad_token_provider = None
        if not aoai_api_key:
            logging.debug("No AZURE_OPENAI_KEY found, using Azure Entra ID auth")
            async with DefaultAzureCredential() as credential:
                ad_token_provider = get_bearer_token_provider(
                    credential,
                    "https://cognitiveservices.azure.com/.default"
                )

        # Deployment
        deployment = app_settings.azure_openai.model
        if not deployment:
            raise ValueError("AZURE_OPENAI_MODEL is required")

        # Default Headers
        default_headers = {"x-ms-useragent": USER_AGENT}

        azure_openai_client = AsyncAzureOpenAI(
            api_version=app_settings.azure_openai.preview_api_version,
            api_key=aoai_api_key,
            azure_ad_token_provider=ad_token_provider,
            default_headers=default_headers,
            azure_endpoint=endpoint,
        )

        return azure_openai_client
    except Exception as e:
        logging.exception("Exception in Azure OpenAI initialization", e)
        azure_openai_client = None
        raise e


async def init_cosmosdb_client():
    cosmos_conversation_client = None
    if app_settings.chat_history:
        try:
            cosmos_endpoint = (
                f"https://{app_settings.chat_history.account}.documents.azure.com:443/"
            )

            if not app_settings.chat_history.account_key:
                async with DefaultAzureCredential() as cred:
                    credential = cred
                    
            else:
                credential = app_settings.chat_history.account_key

            cosmos_conversation_client = CosmosConversationClient(
                cosmosdb_endpoint=cosmos_endpoint,
                credential=credential,
                database_name=app_settings.chat_history.database,
                container_name=app_settings.chat_history.conversations_container,
                enable_message_feedback=app_settings.chat_history.enable_feedback,
            )
        except Exception as e:
            logging.exception("Exception in CosmosDB initialization", e)
            cosmos_conversation_client = None
            raise e
    else:
        logging.debug("CosmosDB not configured")

    return cosmos_conversation_client


def prepare_model_args(request_body, request_headers):
    request_messages = request_body.get("messages", [])
    messages = []
    if not app_settings.datasource:
        messages = [
            {
                "role": "system",
                "content": f"Today's date is {datetime.date.today().isoformat()}. "
                + app_settings.azure_openai.system_message
            }
        ]

    for message in request_messages:
        if message:
            if message["role"] == "assistant" and "context" in message:
                context_obj = json.loads(message["context"])
                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                        "context": context_obj
                    }
                )
            else:
                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }
                )

    user_json = None
    if (MS_DEFENDER_ENABLED):
        authenticated_user_details = get_authenticated_user_details(request_headers)
        conversation_id = request_body.get("conversation_id", None)
        application_name = app_settings.ui.title
        user_json = get_msdefender_user_json(authenticated_user_details, request_headers, conversation_id, application_name)

    model_args = {
        "messages": messages,
        "temperature": app_settings.azure_openai.temperature,
        "max_tokens": app_settings.azure_openai.max_tokens,
        "top_p": app_settings.azure_openai.top_p,
        "stop": app_settings.azure_openai.stop_sequence,
        "stream": app_settings.azure_openai.stream,
        "model": app_settings.azure_openai.model,
        "user": user_json
    }

    if app_settings.datasource:
        model_args["extra_body"] = {
            "data_sources": [
                app_settings.datasource.construct_payload_configuration(
                    request=request
                )
            ]
        }
        # Stamp today's date into the system prompt (role_information) so the model can
        # reason about "next / upcoming / recent / this year" instead of treating an old
        # indexed date as current. The env-var system message stays the static base text.
        _params = model_args["extra_body"]["data_sources"][0].get("parameters", {})
        for _k in ("role_information", "roleInformation"):
            if _params.get(_k):
                _params[_k] = f"Today's date is {datetime.date.today().isoformat()}. " + _params[_k]

    model_args_clean = copy.deepcopy(model_args)
    if model_args_clean.get("extra_body"):
        secret_params = [
            "key",
            "connection_string",
            "embedding_key",
            "encoded_api_key",
            "api_key",
        ]
        for secret_param in secret_params:
            if model_args_clean["extra_body"]["data_sources"][0]["parameters"].get(
                secret_param
            ):
                model_args_clean["extra_body"]["data_sources"][0]["parameters"][
                    secret_param
                ] = "*****"
        authentication = model_args_clean["extra_body"]["data_sources"][0][
            "parameters"
        ].get("authentication", {})
        for field in authentication:
            if field in secret_params:
                model_args_clean["extra_body"]["data_sources"][0]["parameters"][
                    "authentication"
                ][field] = "*****"
        embeddingDependency = model_args_clean["extra_body"]["data_sources"][0][
            "parameters"
        ].get("embedding_dependency", {})
        if "authentication" in embeddingDependency:
            for field in embeddingDependency["authentication"]:
                if field in secret_params:
                    model_args_clean["extra_body"]["data_sources"][0]["parameters"][
                        "embedding_dependency"
                    ]["authentication"][field] = "*****"

    logging.debug(f"REQUEST BODY: {json.dumps(model_args_clean, indent=4)}")

    return model_args


async def promptflow_request(request):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {app_settings.promptflow.api_key}",
        }
        # Adding timeout for scenarios where response takes longer to come back
        logging.debug(f"Setting timeout to {app_settings.promptflow.response_timeout}")
        async with httpx.AsyncClient(
            timeout=float(app_settings.promptflow.response_timeout)
        ) as client:
            pf_formatted_obj = convert_to_pf_format(
                request,
                app_settings.promptflow.request_field_name,
                app_settings.promptflow.response_field_name
            )
            # NOTE: This only support question and chat_history parameters
            # If you need to add more parameters, you need to modify the request body
            response = await client.post(
                app_settings.promptflow.endpoint,
                json={
                    app_settings.promptflow.request_field_name: pf_formatted_obj[-1]["inputs"][app_settings.promptflow.request_field_name],
                    "chat_history": pf_formatted_obj[:-1],
                },
                headers=headers,
            )
        resp = response.json()
        resp["id"] = request["messages"][-1]["id"]
        return resp
    except Exception as e:
        logging.error(f"An error occurred while making promptflow_request: {e}")


# Domain routing (on-your-data fallback). Codes were merged into the code-pipeline index and are
# answered by website_pipeline as 'website', so only permit records may route here (agent off).
PERMITS_INDEX = os.environ.get("AZURE_SEARCH_INDEX_PERMITS")
CODES_INDEX = os.environ.get("AZURE_SEARCH_INDEX_CODES")
# Routing is OPT-IN: active only when BOTH domain indexes are configured. Apps without
# these env vars behave exactly as before, no classifier call, no rerouting.
INDEX_ROUTING_ENABLED = bool(PERMITS_INDEX and CODES_INDEX)

ROUTER_SYSTEM_MESSAGE = (
    "You route a resident's question for a city government assistant to ONE data source. "
    "Reply with exactly one lowercase word: website, permit, events, zoning, or arrest.\n"
    "- arrest: a request to SEE, view, get, or list the arrest log, arrest logs, booking log, or "
    "the police daily arrest log, including for a specific day, week, month, or date range (e.g. "
    "'show me the arrest log for last week', 'yesterday's arrests', 'arrest log for 09-01'). This "
    "is ONLY for the daily arrest/booking log itself. How to file a police report, crime stats, or "
    "general police-department contact are NOT arrest -> route those to website.\n"
    "- website: people, officials, departments, contacts, phone/email, hours, addresses, "
    "city services, news, FAQs, general how-to questions, the municipal code / ordinances / "
    "zoning / regulations themselves (what the code or law says), AND how to apply for or "
    "pay for a permit, permit fees, what documents are needed, which permit you need for a "
    "project, what permit types the city offers in general, and Building & Safety info. "
    'Examples: "how do I apply for a building permit", "how to apply for a permit online", '
    '"what permit do I need for a fence", "what are the permit fees", "what documents do I '
    'need for a permit".\n'
    "- permit: looking up SPECIFIC existing permit records, their status, or any COUNT, "
    "BREAKDOWN, LIST, or RANKING of permits actually filed or issued (this also covers "
    "business tax registrations and business licenses). Includes breakdowns by type, "
    "status, or department, totals over a time period (a year/month), and 'which type or "
    "department has the most'. Examples: 'permit history for 150 N Third St', 'how many "
    "solar permits in December', 'breakdown of permit types in 2025', 'which department "
    "issued the most permits this year', 'how many new businesses opened in 2025'. Use for "
    "existing permit records and their aggregates, NOT for how to apply, fees, or what "
    "permit types exist in general.\n"
    "- events: WHEN something is on the City calendar, its date, time, or schedule. Upcoming City "
    "events, activities, festivals, programs, workshops, things to do, AND City Council / board / "
    "commission MEETING dates and times (e.g. 'when is the next city council meeting', 'next "
    "planning commission meeting'), 'what's happening', 'this weekend', the events or meetings "
    "calendar. This covers only WHAT is scheduled and WHEN. A question about the DETAILS or "
    "LOGISTICS of an event (street or road closures, parking, traffic, routes, rules, how to take "
    "part) is NOT a calendar lookup, route those to website. NOT permit or code lookups.\n"
    "- zoning: use ONLY when the resident names a SPECIFIC PROPERTY or STREET ADDRESS, or supplies a "
    "zoning designation. Examples: 'can I open a medical office at 2019 W Magnolia', 'what can I "
    "build at 123 N Main St', 'is a duplex allowed at [address]', and short follow-ups giving a "
    "designation ('zoning is C-3', 'it's C-3', or just 'C-3', 'R-1', 'MDC-3') answering an earlier "
    "property question. A GENERAL rules question with NO specific property or address is NOT zoning "
    "-> route it to website: e.g. 'can a dumpster be put in an alley', 'how tall can a fence be', "
    "'what are the setback requirements', 'do I need a permit for a shed'. The trigger is a named "
    "property/address or a zoning designation; without one, it is website.\n"
    "If you are unsure, answer website."
)


async def classify_domain(user_query, client, history=None):
    """Return 'website' | 'permit' | 'events'. Defaults to website on any failure.

    If `history` (recent user/assistant turns, ending with the current question) is given,
    the classifier sees it so a short follow-up like 'at what locations?' inherits the topic
    of the previous turn instead of being misread as a generic website question."""
    if not user_query:
        return "website"
    if history:
        system = ROUTER_SYSTEM_MESSAGE + (
            "\nThis is a multi-turn chat. Classify the user's MOST RECENT message, using the "
            "earlier turns only as context. A short follow-up ('at what locations?', 'and in "
            "2024?', 'what about commercial?') inherits the topic of the previous question.")
        convo = [{"role": "system", "content": system}] + history
    else:
        convo = [{"role": "system", "content": ROUTER_SYSTEM_MESSAGE},
                 {"role": "user", "content": user_query}]
    try:
        resp = await client.chat.completions.create(
            model=app_settings.azure_openai.model,
            messages=convo,
            temperature=0,
            max_tokens=5,
        )
        label = (resp.choices[0].message.content or "").strip().lower()
    except Exception:
        logging.exception("Domain classifier failed; defaulting to website")
        return "website"
    if "arrest" in label:
        return "arrest" if ARREST_ROUTE_ENABLED else "website"   # off -> code pipeline, as before
    if "zoning" in label:
        return "zoning" if ZONING_ROUTE_ENABLED else "website"   # off -> code pipeline, as before
    if "permit" in label:
        return "permit"
    if "event" in label:
        return "events"
    return "website"    # website also covers general municipal-code / ordinance questions


def route_index(domain):
    """Map an ALREADY-classified domain to a scoped on-your-data index, or None to keep the default.

    Codes were merged into the code-pipeline index (answered by website_pipeline as 'website'),
    so they are no longer routed here. When the permit AGENT is on, permits are handled by it, so
    in the common config this returns None for everything.
    """
    if domain == "permit" and not PERMIT_AGENT_ENABLED:
        return PERMITS_INDEX
    return None


async def classify_request(request_body, query=None):
    """Classify the question ONCE (website | permit | events) so the permit, events, website,
    and index-routing paths share a single classifier call instead of each making their own. Returns
    None (skip classifying) when no routable feature is enabled or there is no question.
    `query` is the reformulated standalone question (already history-resolved), so no history is
    passed to the classifier; falls back to the raw latest message if not provided."""
    if not (PERMIT_AGENT_ENABLED or EVENTS_ENABLED or CODE_PIPELINE_ENABLED
            or INDEX_ROUTING_ENABLED or ZONING_ROUTE_ENABLED or ARREST_ROUTE_ENABLED):
        return None
    messages = [m for m in request_body.get("messages", []) if m.get("role") != "tool"]
    user_query = query or _latest_user_query(messages)
    if not user_query:
        return None
    try:
        client = await init_openai_client()
        return await classify_domain(user_query, client)
    except Exception:
        logging.exception("domain classification failed; defaulting to website")
        return "website"


# --- Permit agent: answer existing-permit questions from the live records ---------
# Opt-in (askburbanktest only for now). When on, a question the classifier tags as
# 'permit' is answered by the read-permits agent (counts/lists/lookups over the permits
# index) instead of RAG. Everything else (website, codes) is unchanged.
PERMIT_AGENT_ENABLED = bool(permit_agent) and os.environ.get("PERMIT_AGENT_ENABLED", "0") != "0"


def _latest_user_query(messages):
    return next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        None,
    )


def _recent_history(messages, turns=6, max_chars=700):
    """Last few user/assistant turns (content trimmed), so the classifier and the agent
    have conversation context for follow-up questions. Ends with the current question."""
    recent = [m for m in messages
              if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    return [{"role": m["role"], "content": m["content"][:max_chars]} for m in recent[-turns:]]


_REFORMULATE_SYSTEM = (
    "You rewrite the user's latest message into ONE standalone question for a city assistant, using "
    "the earlier conversation ONLY to resolve references.\n"
    "- If the latest message is a genuine follow-up that leans on an earlier turn (e.g. 'what about "
    "in 2024?', 'and commercial ones?', or just a zoning designation like 'C-3'), rewrite it into a "
    "full self-contained question by pulling in the needed context from that earlier turn.\n"
    "- If the latest message is already self-contained, or is a NEW topic unrelated to the earlier "
    "turns, output it unchanged and IGNORE the earlier turns entirely (do not carry over their "
    "subject).\n"
    "Output only the resulting question, nothing else."
)


async def reformulate_query(request_body):
    """History-aware query rewrite, run ONCE upstream of the router. Resolves a genuine follow-up
    from the recent turns, but keeps a new/unrelated question standalone so a prior topic can't bleed
    into retrieval or routing. Returns the query string (the raw latest message on the first turn or
    on any failure). Downstream classify/retrieval all run on this."""
    messages = [m for m in request_body.get("messages", []) if m.get("role") != "tool"]
    latest = _latest_user_query(messages)
    hist = _recent_history(messages)
    prior = hist[:-1] if hist else []
    if not latest or not prior:   # nothing earlier to resolve against
        return latest
    try:
        client = await init_openai_client()
        resp = await client.chat.completions.create(
            model=app_settings.azure_openai.model, temperature=0, max_tokens=120,
            messages=[{"role": "system", "content": _REFORMULATE_SYSTEM}] + prior
                     + [{"role": "user", "content": f"Latest message: {latest}\n\nStandalone question:"}])
        rewritten = (resp.choices[0].message.content or "").strip()
        if rewritten and rewritten != latest:
            logging.info("[REFORMULATE] %r -> %r", latest, rewritten)
        return rewritten or latest
    except Exception:
        logging.exception("query reformulation failed; using the raw latest message")
        return latest


async def try_permit_answer(request_body, domain, query=None):
    """If the question is a permit-records question, answer it from the live permits index and return
    the answer string. Otherwise return None (run normal RAG). `query` is the reformulated standalone
    question; the raw recent history is still passed to the agent for its tool loop."""
    if not PERMIT_AGENT_ENABLED or domain != "permit":
        return None
    messages = [m for m in request_body.get("messages", []) if m.get("role") != "tool"]
    user_query = query or _latest_user_query(messages)
    if not user_query:
        return None
    try:
        client = await init_openai_client()
        history = _recent_history(messages)
        logging.info("[PERMIT AGENT] handling: %s", user_query)
        return await permit_agent.answer_permit_query(
            user_query, client, app_settings.azure_openai.model, history=history)
    except Exception:
        logging.exception("permit agent failed; falling back to RAG")
        return None


# City calendar lookup (events + meetings) from events.json. Routed via the classifier's
# 'events' domain. Replaces the Granicus meetings feed; events.json must be kept fresh.
EVENTS_ENABLED = bool(events_feed) and events_feed.available()


async def try_events_answer(request_body, domain, query=None):
    """If the classifier says EVENTS (upcoming events or a Council/board/commission meeting time),
    answer from the current events.json. Returns the answer string, or None to fall through to RAG.
    `query` is the reformulated standalone question."""
    if not EVENTS_ENABLED or domain != "events":
        return None
    try:
        client = await init_openai_client()
        user_query = query or _latest_user_query(request_body.get("messages", []))
        if not user_query:
            return None
        logging.info("[EVENTS] handling: %s", user_query)
        return await events_feed.answer_events_query(
            user_query, client, app_settings.azure_openai.model, datetime.date.today())
    except Exception:
        logging.exception("events feed failed; falling back to RAG")
        return None


def _permit_message_obj(msg_id=None):
    # unique per answer: the frontend echoes this response id as the message id (Chat.tsx),
    # so a constant here makes every answer collide on one message doc (breaks feedback + history).
    return {
        "id": msg_id or str(uuid.uuid4()),
        "model": app_settings.azure_openai.model,
        "created": int(time.time()),
        "object": "extensions.chat.completion",
        "choices": [{"messages": []}],
    }


def permit_non_streaming_response(answer, history_metadata):
    """Shape a permit answer exactly like format_non_streaming_response output."""
    obj = _permit_message_obj()
    obj["choices"][0]["messages"].append({"role": "assistant", "content": answer})
    obj["history_metadata"] = history_metadata
    obj["apim-request-id"] = obj["id"]
    return obj


def permit_stream_response(answer, history_metadata):
    """A one-chunk async stream shaped like format_stream_response output."""
    async def generate():
        obj = _permit_message_obj()
        obj["object"] = "extensions.chat.completion.chunk"
        obj["choices"][0]["messages"].append({"role": "assistant", "content": answer})
        obj["history_metadata"] = history_metadata
        obj["apim-request-id"] = obj["id"]
        yield obj
    return generate()


def _website_messages(answer, context):
    """Tool (citations) + assistant, same shape on-your-data produces, so the frontend renders
    citations identically."""
    return [{"role": "tool", "content": json.dumps(context)},
            {"role": "assistant", "content": answer}]


def website_non_streaming_response(answer, context, history_metadata, answer_id=None):
    obj = _permit_message_obj(answer_id)
    obj["choices"][0]["messages"] = _website_messages(answer, context)
    obj["history_metadata"] = history_metadata
    obj["apim-request-id"] = obj["id"]
    return obj


def website_stream_response(answer, context, history_metadata, answer_id=None):
    async def generate():
        obj = _permit_message_obj(answer_id)
        obj["object"] = "extensions.chat.completion.chunk"
        obj["choices"][0]["messages"] = _website_messages(answer, context)
        obj["history_metadata"] = history_metadata
        obj["apim-request-id"] = obj["id"]
        yield obj
    return generate()


async def try_arrest_answer(request_body, domain):
    """Arrest / daily-log requests -> a FIXED deflect to the live arrest log page. Deterministic
    (no LLM, no retrieval) so it can't enumerate or invent daily-log dates or contents. Returns the
    canned answer string, or None to fall through."""
    if not ARREST_ROUTE_ENABLED or domain != "arrest":
        return None
    logging.info("[ARREST] deflect to %s", ARREST_LOG_URL)
    return ARREST_ANSWER


async def try_zoning_answer(request_body, domain, query=None):
    """Address-specific zoning / land-use questions -> answered from the code index with a zoning
    prompt (asks for the designation if absent, else answers from the code). Returns website_pipeline's
    (answer, context, answer_id) or None to fall through. `query` is the reformulated standalone
    question (a bare 'C-3' follow-up is already resolved upstream, so no concatenation here)."""
    if not ZONING_ROUTE_ENABLED or domain != "zoning":
        return None
    messages = [m for m in request_body.get("messages", []) if m.get("role") != "tool"]
    q = query or _latest_user_query(messages)
    if not q:
        return None
    try:
        client = await init_openai_client()
        logging.info("[ZONING] %s", q)
        return await zoning.answer_zoning_query(q, client, app_settings.azure_openai.model)
    except Exception:
        logging.exception("zoning route failed; falling back")
        return None


async def try_website_answer(request_body, domain, query=None):
    """If enabled and the classifier says WEBSITE, answer from burbank-code-v1 (hybrid + semantic
    rerank + in-depth prompt). Returns (answer, context) or None to fall through to on-your-data.
    Municipal-code questions now classify as WEBSITE and are answered here too; permit/meetings
    never reach here. `query` is the reformulated standalone question."""
    if not CODE_PIPELINE_ENABLED or domain != "website":
        return None
    try:
        client = await init_openai_client()
        q = query or _latest_user_query(request_body.get("messages", []))
        if not q:
            return None
        logging.info("[CODE PIPELINE] website: %s", q)
        # reuse the chat completion's own id (resp.id) as the response/message id, no invented uuid
        return await website_pipeline.answer_website_query(
            q, client, app_settings.azure_openai.model)
    except Exception:
        logging.exception("website pipeline failed; falling back to on-your-data")
        return None


async def send_chat_request(request_body, request_headers, domain=None):
    filtered_messages = []
    messages = request_body.get("messages", [])
    for message in messages:
        if message.get("role") != 'tool':
            filtered_messages.append(message)
            
    request_body['messages'] = filtered_messages
    model_args = prepare_model_args(request_body, request_headers)

    try:
        azure_openai_client = await init_openai_client()

        # Route this question to the right scoped index (website stays default).
        if INDEX_ROUTING_ENABLED and app_settings.datasource and model_args.get("extra_body"):
            routed_index = route_index(domain)
            if routed_index:
                model_args["extra_body"]["data_sources"][0]["parameters"]["index_name"] = routed_index
                logging.info(f"[ROUTED INDEX] {routed_index}")

        raw_response = await azure_openai_client.chat.completions.with_raw_response.create(**model_args)
        response = raw_response.parse()
        apim_request_id = raw_response.headers.get("apim-request-id") 
    except Exception as e:
        logging.exception("Exception in send_chat_request")
        raise e

    return response, apim_request_id


async def complete_chat_request(request_body, request_headers):
    if app_settings.base_settings.use_promptflow:
        response = await promptflow_request(request_body)
        history_metadata = request_body.get("history_metadata", {})
        return format_pf_non_streaming_response(
            response,
            history_metadata,
            app_settings.promptflow.response_field_name,
            app_settings.promptflow.citations_field_name
        )
    else:
        history_metadata = request_body.get("history_metadata", {})
        q = await reformulate_query(request_body)          # history-resolved standalone query, ONCE
        domain = await classify_request(request_body, q)   # classify ONCE; reused by every route
        permit_answer = await try_permit_answer(request_body, domain, q)
        if permit_answer is not None:
            return permit_non_streaming_response(permit_answer, history_metadata)
        events_answer = await try_events_answer(request_body, domain, q)
        if events_answer is not None:
            return permit_non_streaming_response(events_answer, history_metadata)
        arrest_answer = await try_arrest_answer(request_body, domain)
        if arrest_answer is not None:
            return permit_non_streaming_response(arrest_answer, history_metadata)
        zoning_answer = await try_zoning_answer(request_body, domain, q)
        if zoning_answer is not None:
            return website_non_streaming_response(zoning_answer[0], zoning_answer[1], history_metadata, zoning_answer[2])
        website = await try_website_answer(request_body, domain, q)   # website/codes -> code pipeline
        if website is not None:
            return website_non_streaming_response(website[0], website[1], history_metadata, website[2])
        response, apim_request_id = await send_chat_request(request_body, request_headers, domain)
        return format_non_streaming_response(response, history_metadata, apim_request_id)


async def stream_chat_request(request_body, request_headers):
    history_metadata = request_body.get("history_metadata", {})
    q = await reformulate_query(request_body)          # history-resolved standalone query, ONCE
    domain = await classify_request(request_body, q)   # classify ONCE; reused by every route
    permit_answer = await try_permit_answer(request_body, domain, q)
    if permit_answer is not None:
        return permit_stream_response(permit_answer, history_metadata)
    events_answer = await try_events_answer(request_body, domain, q)
    if events_answer is not None:
        return permit_stream_response(events_answer, history_metadata)
    arrest_answer = await try_arrest_answer(request_body, domain)
    if arrest_answer is not None:
        return permit_stream_response(arrest_answer, history_metadata)
    zoning_answer = await try_zoning_answer(request_body, domain, q)
    if zoning_answer is not None:
        return website_stream_response(zoning_answer[0], zoning_answer[1], history_metadata, zoning_answer[2])
    website = await try_website_answer(request_body, domain, q)   # website/codes -> code pipeline
    if website is not None:
        return website_stream_response(website[0], website[1], history_metadata, website[2])
    response, apim_request_id = await send_chat_request(request_body, request_headers, domain)

    async def generate():
        scrubber = BlockedTextScrubber()
        meta = None
        async for completionChunk in response:
            obj = format_stream_response(completionChunk, history_metadata, apim_request_id)
            if not obj:
                continue
            messages = obj.get("choices", [{}])[0].get("messages", [])
            content_msg = next(
                (m for m in messages if m.get("role") == "assistant" and "content" in m),
                None,
            )
            if content_msg is not None:
                meta = {k: obj.get(k) for k in ("id", "model", "created", "object")}
                emitted = scrubber.feed(content_msg["content"])
                if not emitted:
                    continue
                content_msg["content"] = emitted
                yield obj
            else:
                yield obj  # context / citation messages pass through untouched
        tail = scrubber.flush()
        if tail:
            base = meta or {
                "id": "", "model": "", "created": int(time.time()),
                "object": "extensions.chat.completion.chunk",
            }
            yield {
                **base,
                "choices": [{"messages": [{"role": "assistant", "content": tail}]}],
                "history_metadata": history_metadata,
                "apim-request-id": apim_request_id,
            }

    return generate()


async def conversation_internal(request_body, request_headers):
    try:
        if app_settings.azure_openai.stream and not app_settings.base_settings.use_promptflow:
            result = await stream_chat_request(request_body, request_headers)
            response = await make_response(format_as_ndjson(result))
            response.timeout = None
            response.mimetype = "application/json-lines"
            return response
        else:
            result = await complete_chat_request(request_body, request_headers)
            return jsonify(result)

    except Exception as ex:
        logging.exception(ex)
        if hasattr(ex, "status_code"):
            return jsonify({"error": str(ex)}), ex.status_code
        else:
            return jsonify({"error": str(ex)}), 500


@bp.route("/conversation", methods=["POST"])
async def conversation():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()

    return await conversation_internal(request_json, request.headers)


@bp.route("/frontend_settings", methods=["GET"])
def get_frontend_settings():
    try:
        return jsonify(frontend_settings), 200
    except Exception as e:
        logging.exception("Exception in /frontend_settings")
        return jsonify({"error": str(e)}), 500


## Conversation History API ##
@bp.route("/history/generate", methods=["POST"])
async def add_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        history_metadata = {}
        if not conversation_id:
            title = await generate_title(request_json["messages"])
            conversation_dict = await current_app.cosmos_conversation_client.create_conversation(
                user_id=user_id, title=title
            )
            conversation_id = conversation_dict["id"]
            history_metadata["title"] = title
            history_metadata["date"] = conversation_dict["createdAt"]

        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "user":
            createdMessageValue = await current_app.cosmos_conversation_client.create_message(
                uuid=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
            if createdMessageValue == "Conversation not found":
                raise Exception(
                    "Conversation not found for the given conversation ID: "
                    + conversation_id
                    + "."
                )
        else:
            raise Exception("No user message found")

        # Submit request to Chat Completions for response
        request_body = await request.get_json()
        history_metadata["conversation_id"] = conversation_id
        request_body["history_metadata"] = history_metadata
        return await conversation_internal(request_body, request.headers)

    except Exception as e:
        logging.exception("Exception in /history/generate")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/update", methods=["POST"])
async def update_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        if not conversation_id:
            raise Exception("No conversation_id found")

        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "assistant":
            if len(messages) > 1 and messages[-2].get("role", None) == "tool":
                # write the tool message first
                await current_app.cosmos_conversation_client.create_message(
                    uuid=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    user_id=user_id,
                    input_message=messages[-2],
                )
            # write the assistant message
            await current_app.cosmos_conversation_client.create_message(
                uuid=messages[-1]["id"],
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
        else:
            raise Exception("No bot messages found")

        # Submit request to Chat Completions for response
        response = {"success": True}
        return jsonify(response), 200

    except Exception as e:
        logging.exception("Exception in /history/update")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/message_feedback", methods=["POST"])
async def update_message():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for message_id
    request_json = await request.get_json()
    message_id = request_json.get("message_id", None)
    message_feedback = request_json.get("message_feedback", None)    
    other_text = request_json.get("other_text", None)
    
    try:
        if not message_id:
            return jsonify({"error": "message_id is required"}), 400

        if not message_feedback:
            return jsonify({"error": "message_feedback is required"}), 400

        ## update the message in cosmos
        updated_message = await current_app.cosmos_conversation_client.update_message_feedback(
            user_id, message_id, message_feedback, other_text
        )
        if updated_message:
            return (
                jsonify(
                    {
                        "message": f"Successfully updated message with feedback {message_feedback}",
                        "message_id": message_id,
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "error": f"Unable to update message {message_id}. It either does not exist or the user does not have access to it."
                    }
                ),
                404,
            )

    except Exception as e:
        logging.exception("Exception in /history/message_feedback")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/delete", methods=["DELETE"])
async def delete_conversation():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400

        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos first
        deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
            conversation_id, user_id
        )

        ## Now delete the conversation
        deleted_conversation = await current_app.cosmos_conversation_client.delete_conversation(
            user_id, conversation_id
        )

        return (
            jsonify(
                {
                    "message": "Successfully deleted conversation and messages",
                    "conversation_id": conversation_id,
                }
            ),
            200,
        )
    except Exception as e:
        logging.exception("Exception in /history/delete")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/list", methods=["GET"])
async def list_conversations():
    await cosmos_db_ready.wait()
    offset = request.args.get("offset", 0)
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversations from cosmos
    conversations = await current_app.cosmos_conversation_client.get_conversations(
        user_id, offset=offset, limit=25
    )
    if not isinstance(conversations, list):
        return jsonify({"error": f"No conversations for {user_id} were found"}), 404

    ## return the conversation ids

    return jsonify(conversations), 200


@bp.route("/history/read", methods=["POST"])
async def get_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversation object and the related messages from cosmos
    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    ## return the conversation id and the messages in the bot frontend format
    if not conversation:
        return (
            jsonify(
                {
                    "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
                }
            ),
            404,
        )

    # get the messages for the conversation from cosmos
    conversation_messages = await current_app.cosmos_conversation_client.get_messages(
        user_id, conversation_id
    )

    ## format the messages in the bot frontend format
    messages = [
        {
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"],
            "createdAt": msg["createdAt"],
            "feedback": msg.get("feedback"),
        }
        for msg in conversation_messages
    ]

    return jsonify({"conversation_id": conversation_id, "messages": messages}), 200


@bp.route("/history/rename", methods=["POST"])
async def rename_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversation from cosmos
    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    if not conversation:
        return (
            jsonify(
                {
                    "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
                }
            ),
            404,
        )

    ## update the title
    title = request_json.get("title", None)
    if not title:
        return jsonify({"error": "title is required"}), 400
    conversation["title"] = title
    updated_conversation = await current_app.cosmos_conversation_client.upsert_conversation(
        conversation
    )

    return jsonify(updated_conversation), 200


@bp.route("/history/delete_all", methods=["DELETE"])
async def delete_all_conversations():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    # get conversations for user
    try:
        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        conversations = await current_app.cosmos_conversation_client.get_conversations(
            user_id, offset=0, limit=None
        )
        if not conversations:
            return jsonify({"error": f"No conversations for {user_id} were found"}), 404

        # delete each conversation
        for conversation in conversations:
            ## delete the conversation messages from cosmos first
            deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
                conversation["id"], user_id
            )

            ## Now delete the conversation
            deleted_conversation = await current_app.cosmos_conversation_client.delete_conversation(
                user_id, conversation["id"]
            )
        return (
            jsonify(
                {
                    "message": f"Successfully deleted conversation and messages for user {user_id}"
                }
            ),
            200,
        )

    except Exception as e:
        logging.exception("Exception in /history/delete_all")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/clear", methods=["POST"])
async def clear_messages():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400

        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos
        deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
            conversation_id, user_id
        )

        return (
            jsonify(
                {
                    "message": "Successfully deleted messages in conversation",
                    "conversation_id": conversation_id,
                }
            ),
            200,
        )
    except Exception as e:
        logging.exception("Exception in /history/clear_messages")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/ensure", methods=["GET"])
async def ensure_cosmos():
    await cosmos_db_ready.wait()
    if not app_settings.chat_history:
        return jsonify({"error": "CosmosDB is not configured"}), 404

    try:
        success, err = await current_app.cosmos_conversation_client.ensure()
        if not current_app.cosmos_conversation_client or not success:
            if err:
                return jsonify({"error": err}), 422
            return jsonify({"error": "CosmosDB is not configured or not working"}), 500

        return jsonify({"message": "CosmosDB is configured and working"}), 200
    except Exception as e:
        logging.exception("Exception in /history/ensure")
        cosmos_exception = str(e)
        if "Invalid credentials" in cosmos_exception:
            return jsonify({"error": cosmos_exception}), 401
        elif "Invalid CosmosDB database name" in cosmos_exception:
            return (
                jsonify(
                    {
                        "error": f"{cosmos_exception} {app_settings.chat_history.database} for account {app_settings.chat_history.account}"
                    }
                ),
                422,
            )
        elif "Invalid CosmosDB container name" in cosmos_exception:
            return (
                jsonify(
                    {
                        "error": f"{cosmos_exception}: {app_settings.chat_history.conversations_container}"
                    }
                ),
                422,
            )
        else:
            return jsonify({"error": "CosmosDB is not working"}), 500


async def generate_title(conversation_messages) -> str:
    ## make sure the messages are sorted by _ts descending
    title_prompt = "Summarize the conversation so far into a 4-word or less title. Do not use any quotation marks or punctuation. Do not include any other commentary or description."

    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation_messages
    ]
    messages.append({"role": "user", "content": title_prompt})

    try:
        azure_openai_client = await init_openai_client()
        response = await azure_openai_client.chat.completions.create(
            model=app_settings.azure_openai.model, messages=messages, temperature=1, max_tokens=64
        )

        title = response.choices[0].message.content
        return title
    except Exception as e:
        logging.exception("Exception while generating title", e)
        return messages[-2]["content"]


app = create_app()
