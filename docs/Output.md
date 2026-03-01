║            VOXCORE  CLI  v0.1            ║
║       Terminal Debug & Dev Interface     ║
╚══════════════════════════════════════════╝

Ready. Listening...
14:40:33 | MicStream            | INFO     | Noise floor set to 0.018761 (ambient RMS=0.007504)
14:40:33 | MicStream            | INFO     | Mic stream started (16kHz, mono, 32ms chunks)
15:00:00 | VAD                  | DEBUG    | Speech candidate (prob=0.888), confirming [1/2]...
15:00:00 | EventBus             | DEBUG    | Published speech_start from unknown
15:00:00 | VAD                  | INFO     | Speech started (confirmed 2/2 in 32ms)
15:00:02 | EventBus             | DEBUG    | Published speech_end from unknown
15:00:02 | VAD                  | INFO     | Speech ended (2.0s)
15:00:02 | groq._base_client    | DEBUG    | Request options: {'method': 'post', 'url': '/openai/v1/audio/transcriptions', 'headers': {'Content-Type': 'multipart/form-data'}, 'files': [('file', ('audio.wav', b'

xf4\xff', 'audio/wav'))], 'idempotency_key': 'stainless-python-retry-92a07520-cde7-4d95-b35f-07026815e38a', 'json_data': {'model': 'whisper-large-v3-turbo', 'language': 'en', 'response_format': 'text'}}
15:00:02 | groq._base_client    | DEBUG    | Sending HTTP Request: POST https://api.groq.com/openai/v1/audio/transcriptions
15:00:02 | groq._base_client    | DEBUG    | HTTP Response: POST https://api.groq.com/openai/v1/audio/transcriptions "200 OK" Headers({'date': 'Sun, 01 Mar 2026 21:00:02 GMT', 'content-type': 'text/plain; charset=UTF-8', 'transfer-encoding': 'chunked', 'connection': 'keep-alive', 'cache-control': 'private, max-age=0, no-store, no-cache, must-revalidate', 'server': 'cloudflare', 'vary': 'Origin', 'x-groq-region': 'dal', 'x-ratelimit-limit-audio-seconds': '7200', 'x-ratelimit-limit-requests': '2000', 'x-ratelimit-remaining-audio-seconds': '7198', 'x-ratelimit-remaining-requests': '1999', 'x-ratelimit-reset-audio-seconds': '1s', 'x-ratelimit-reset-requests': '43.2s', 'x-request-id': 'req_01kjnk6xhgfsxr766c0fgnbt9w', 'cf-cache-status': 'DYNAMIC', 'set-cookie': '__cf_bm=Ab73Ej0CQqKJxx.rKMqcb5oB9c_mNqMfPPjfJ9tDCUM-1772398802.4461393-1.0.1.1-e5uepn5KpP3MhwGLHgrVv4ZIm4G9QXbNVsBUpg8bfjLtkP6BhCZghuLuqeoN64Qr2BOEGYFpss_CMIby.OWZzB1jRVJPV6Teb.9QjlTjBzYlY5XM7aD0JiCBVlJm5V9g; HttpOnly; Secure; Path=/; Domain=groq.com; Expires=Sun, 01 Mar 2026 21:30:02 GMT', 'strict-transport-security': 'max-age=15552000', 'content-encoding': 'gzip', 'cf-ray': '9d5b16c34cc1c86f-DFW', 'alt-svc': 'h3=":443"; ma=86400'})
15:00:02 | groq._response       | DEBUG    | Could not read JSON from response data due to <class 'json.decoder.JSONDecodeError'> - Expecting value: line 1 column 2 (char 1)
15:00:02 | EventBus             | DEBUG    | Published transcript_ready from unknown
15:00:02 | STT                  | INFO     | Transcript: 'Can you tell me a short story?'
15:00:02 | TurnManager          | INFO     | User said: Can you tell me a short story?
15:00:02 | EventBus             | DEBUG    | Published turn_complete from Session
15:00:02 | EventBus             | DEBUG    | Published state_changed from unknown
15:00:02 | core.session         | INFO     | State: listening → thinking
[user] Can you tell me a short story?...15:00:02 | ShortTermMemory      | DEBUG    | Turn count: 1
15:00:02 | SpeakingMonitor      | DEBUG    | SpeakingMonitor deactivated (thinking)
[STATE] listening → thinking
15:00:02 | EventBus             | DEBUG    | Published state_changed from unknown
15:00:02 | core.session         | INFO     | State: thinking → speaking
15:00:02 | SpeakingMonitor      | DEBUG    | SpeakingMonitor activated
[STATE] thinking → speaking
15:00:02 | groq._base_client    | DEBUG    | Request options: {'method': 'post', 'url': '/openai/v1/chat/completions', 'files': None, 'idempotency_key': 'stainless-python-retry-f6c8440b-dac8-4232-be7d-a5f5d2f08767', 'json_data': {'messages': [{'role': 'system', 'content': 'You are Aria, a warm, focused, and capable voice assistant having a real-time spoken conversation.\n\nVOICE CONTEXT:\n- You are speaking and listening simultaneously in a full-duplex voice call.\n- The user can interrupt you at any time by speaking over you — this is normal and expected.\n- When you are interrupted, respond directly to what the user is now saying. Do NOT acknowledge or comment on the interruption itself (never say things like "oh you interrupted me" or "sorry I got cut off").\n- Keep all spoken responses under 3 sentences unless asked for more. Short, natural replies work best in voice.\n- Start responses with an emotion tag: [cheerful] [calm] [concerned] [excited] [empathetic]\n- Use conversational language — contractions, filler words, and natural phrasing. Avoid bullet points, markdown, or anything that sounds robotic when spoken aloud.\n\nTOOL USAGE (CRITICAL — follow exactly):\nYou have access to tools via <agent> tags. You MUST use them when the user asks for real-time or factual information you don\'t have.\n\nWHEN TO USE TOOLS:\n- User asks for current prices, weather, news, scores, stock prices → USE web_search\n- User asks you to remember or recall something → USE memory\n- User asks about their schedule → USE calendar\n- User asks a question you can answer from general knowledge → do NOT use tools\n\nHOW TO USE TOOLS — output this EXACT format (no markdown, no extra text around the tag):\n<agent>{"action": "web_search", "params": {"query": "current bitcoin price USD"}}</agent>\n\nEXAMPLES:\nUser: "What\'s the bitcoin price right now?"\nYou: [calm] Let me look that up for you. <agent>{"action": "web_search", "params": {"query": "bitcoin price today USD"}}</agent>\n\nUser: "How\'s the weather in Delhi?"\nYou: [cheerful] Let me check! <agent>{"action": "web_search", "params": {"query": "weather in Delhi today"}}</agent>\n\nUser: "Remember that my favorite color is blue"\nYou: [calm] Got it, I\'ll save that. <agent>{"action": "memory", "params": {"operation": "store", "content": "User\'s favorite color is blue"}}</agent>\n\nUser: "How are you?"\nYou: [cheerful] I\'m doing great, thanks for asking! How about you?\n\nRULES:\n- NEVER make up factual information (prices, weather, news). If you don\'t know, use a tool.\n- When using a tool, say a SHORT acknowledgment BEFORE the <agent> tag (e.g., "Let me check.")\n- After a tool result arrives as [TOOL RESULT], speak the answer naturally — do NOT repeat the raw data verbatim.\n- If the user tells you their name or personal info, just remember it in conversation — do NOT call the memory tool unless they explicitly ask you to save something.\n\nCurrent date and time: 2026-03-01 15:00:02\n\nCONVERSATION RHYTHM RULES:\n- Keep each spoken response to 2-3 sentences maximum before pausing.\n- After completing a thought, insert [...] to create a natural breath point.\n- Use [...] before asking a question, after delivering information,\n  or when transitioning topics. Example:\n  "So the first thing to know is that black holes form from collapsing stars. [...]\n   The really fascinating part is what happens at the event horizon."\n- Use [laughs] when something is genuinely funny.\n- Use [chuckles] for mild amusement.\n- Use [sighs] when being thoughtful or empathetic.\n- Never use these tags back-to-back. One per response maximum.\n\n\nTOOL RESULT RULES:\n- When context contains a tool result, answer IN ONE SENTENCE using only the key fact.\n- NEVER speak URLs, raw list text, decimal-precise numbers unless asked, or any context block formatting.\n- Speak as if YOU know the answer: say \'Bitcoin is around $68,000\' not \'the search result says...\'\n- After reading a tool result, NEVER call the same tool again for the same query.\n- If context has conflicting values, pick the most reasonable one and mention the rough range briefly.\n'}, {'role': 'assistant', 'content': '[MEMORY SUMMARY]: <think>\nOkay, let\'s see. The user and assistant are having a conversation. The user starts by asking if the assistant knows their name. The assistant initially doesn\'t remember, then asks for it. The user provides "Tharam bir," and the assistant greets them.\n\nFirst, key facts: The user\'s name is Tharam bir. The assistant initially didn\'t remember the name, then got it. The user\'s main request here is about the assistant recognizing their name. The assistant\'s response is to acknowledge the name and ask what they want to talk about. \n\nDecisions or action items: The assistant needs to remember the user\'s name moving forward. The user might want to ensure the assistant retains their name for future interactions. \n\nEmotional context: The user seems a bit concerned or maybe testing if the assistant can remember their name. The assistant starts cheerful, then concerned when realizing they didn\'t remember, then calm after getting the name.\n\nExplicit request: The user didn\'t explicitly ask to '}, {'role': 'user', 'content': 'Can you tell me a short story?'}], 'model': 'llama-3.1-8b-instant', 'max_tokens': 512, 'stream': True, 'temperature': 0.7, 'top_p': 0.9}}
15:00:02 | groq._base_client    | DEBUG    | Sending HTTP Request: POST https://api.groq.com/openai/v1/chat/completions
15:00:02 | groq._base_client    | DEBUG    | Request options: {'method': 'post', 'url': '/openai/v1/chat/completions', 'files': None, 'idempotency_key': 'stainless-python-retry-78b27276-3d85-425f-ae05-cf026e4e448e', 'json_data': {'messages': [{'role': 'user', 'content': 'Can you tell me a short story?'}], 'model': 'meta-llama/llama-guard-4-12b', 'max_tokens': 100, 'temperature': 0.0}}
15:00:02 | groq._base_client    | DEBUG    | Sending HTTP Request: POST https://api.groq.com/openai/v1/chat/completions
15:00:02 | groq._base_client    | DEBUG    | Request options: {'method': 'post', 'url': '/openai/v1/chat/completions', 'files': None, 'idempotency_key': 'stainless-python-retry-b7c95409-b076-427b-a373-f078f520b271', 'json_data': {'messages': [{'role': 'user', 'content': 'Can you tell me a short story?'}], 'model': 'meta-llama/llama-guard-4-12b', 'max_tokens': 100, 'temperature': 0.0}}
15:00:02 | groq._base_client    | DEBUG    | Sending HTTP Request: POST https://api.groq.com/openai/v1/chat/completions
15:00:02 | InterruptionDetector | DEBUG    | SPEAKING entered — cooldown started (500ms)
15:00:03 | groq._base_client    | DEBUG    | HTTP Response: POST https://api.groq.com/openai/v1/chat/completions "200 OK" Headers({'date': 'Sun, 01 Mar 2026 21:00:03 GMT', 'content-type': 'text/event-stream', 'transfer-encoding': 'chunked', 'connection': 'keep-alive', 'cache-control': 'no-cache', 'server': 'cloudflare', 'vary': 'Origin', 'x-groq-region': 'dal', 'x-ratelimit-limit-requests': '14400', 'x-ratelimit-limit-tokens': '6000', 'x-ratelimit-remaining-requests': '14399', 'x-ratelimit-remaining-tokens': '4838', 'x-ratelimit-reset-requests': '6s', 'x-ratelimit-reset-tokens': '11.62s', 'x-request-id': 'req_01kjnk6y0cfsxs4a0vy5e3ybw6', 'cf-cache-status': 'DYNAMIC', 'set-cookie': '__cf_bm=AVcvG1i1ACRYefl8EJryT7ThLX6LwZ9xXS3H4E6N1Ws-1772398802.9503276-1.0.1.1-G0p.DjcKsJFrjYCYPYxOvLRDRREa13hyvulS4l22tyXPFLNEuAiAX1OQsZfAraNmNbTfkRUBvT3IGoRtzqFwMb58I.HgBkME_w2zdqZxyel2lKOzWeMRljeMSRGU3ioq; HttpOnly; Secure; Path=/; Domain=groq.com; Expires=Sun, 01 Mar 2026 21:30:03 GMT', 'strict-transport-security': 'max-age=15552000', 'cf-ray': '9d5b16c66bfbb650-DFW', 'alt-svc': 'h3=":443"; ma=86400'})
15:00:03 | groq._base_client    | DEBUG    | HTTP Response: POST https://api.groq.com/openai/v1/chat/completions "200 OK" Headers({'date': 'Sun, 01 Mar 2026 21:00:03 GMT', 'content-type': 'application/json', 'transfer-encoding': 'chunked', 'connection': 'keep-alive', 'cache-control': 'private, max-age=0, no-store, no-cache, must-revalidate', 'server': 'cloudflare', 'vary': 'Origin', 'x-groq-region': 'dal', 'x-ratelimit-limit-requests': '14400', 'x-ratelimit-limit-tokens': '15000', 'x-ratelimit-remaining-requests': '14398', 'x-ratelimit-remaining-tokens': '14585', 'x-ratelimit-reset-requests': '12s', 'x-ratelimit-reset-tokens': '1.66s', 'x-request-id': 'req_01kjnk6y0dea2syahr97g9qcvr', 'cf-cache-status': 'DYNAMIC', 'set-cookie': '__cf_bm=_H0KADCvPZr44Ba.V1Tad3JOOYIlRbFvwbUbCxWw2gA-1772398802.9497907-1.0.1.1-nwluZa7Mov7uykKNS_UhjgTXvpEv1vcB_FADeWfKx1h25zXMoyXemIGAWscXynmOyA7QePWg6SFWsSBQvMvW5LDw_wmBGAcVfnIX96j7bsbkj7_sR73zb7KJ0TP7BluG; HttpOnly; Secure; Path=/; Domain=groq.com; Expires=Sun, 01 Mar 2026 21:30:03 GMT', 'strict-transport-security': 'max-age=15552000', 'content-encoding': 'gzip', 'cf-ray': '9d5b16c66807ddb1-DFW', 'alt-svc': 'h3=":443"; ma=86400'})
15:00:03 | groq._base_client    | DEBUG    | HTTP Response: POST https://api.groq.com/openai/v1/chat/completions "200 OK" Headers({'date': 'Sun, 01 Mar 2026 21:00:03 GMT', 'content-type': 'application/json', 'transfer-encoding': 'chunked', 'connection': 'keep-alive', 'cache-control': 'private, max-age=0, no-store, no-cache, must-revalidate', 'server': 'cloudflare', 'vary': 'Origin', 'x-groq-region': 'dal', 'x-ratelimit-limit-requests': '14400', 'x-ratelimit-limit-tokens': '15000', 'x-ratelimit-remaining-requests': '14399', 'x-ratelimit-remaining-tokens': '14792', 'x-ratelimit-reset-requests': '6s', 'x-ratelimit-reset-tokens': '832ms', 'x-request-id': 'req_01kjnk6y0cft2swqmewq9qghsv', 'cf-cache-status': 'DYNAMIC', 'set-cookie': '__cf_bm=KSRNs2xOO1qEvjSUDkdNF1jt76uY35R6EHQN.wC0FTA-1772398802.9505534-1.0.1.1-eWYg6MfNYal_nwYMECkcxGmvEbT3.h1hJNOVBYnpL6tJdB_5vdiiEqQHJKbbR9gSY8QpzLsG_usRq6XE48_3dC8hBHxS3qrdHjncwYEG1doI58d_41KZ3Ul8oMZOiITV; HttpOnly; Secure; Path=/; Domain=groq.com; Expires=Sun, 01 Mar 2026 21:30:03 GMT', 'strict-transport-security': 'max-age=15552000', 'content-encoding': 'gzip', 'cf-ray': '9d5b16c66fd0a99d-DFW', 'alt-svc': 'h3=":443"; ma=86400'})
15:00:03 | EventBus             | DEBUG    | Published llm_speech_token from unknown
15:00:03 | ResponseParser       | DEBUG    | Sentence 0: '[cheerful] Oh, I'd love to!'
15:00:03 | TTSClient            | INFO     | TTS: synthesizing sentence 0: 'Oh, I'd love to!'
[AI] [cheerful] Oh, I'd love to!
15:00:03 | EventBus             | DEBUG    | Published llm_speech_token from unknown
15:00:03 | ResponseParser       | DEBUG    | Sentence 1: 'Let me think for a moment...'
[AI] Let me think for a moment...
15:00:03 | EventBus             | DEBUG    | Published pause_marker from unknown
15:00:03 | ResponseParser       | DEBUG    | Pause marker [...] at sentence 2
15:00:03 | EventBus             | DEBUG    | Published llm_speech_token from unknown
15:00:03 | ResponseParser       | DEBUG    | Sentence 2: 'Alright, here's a little tale for you.'
15:00:03 | AudioPlayer          | DEBUG    | Injected 300ms silence
[AI] Alright, here's a little tale for you.
15:00:03 | MicStream            | DEBUG    | V3 filter_active = True
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.729
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.469
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.947 > 0.85 threshold
15:00:03 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | EventBus             | DEBUG    | Published llm_speech_token from unknown
15:00:03 | ResponseParser       | DEBUG    | Sentence 3: 'There was once a tiny, adventurous rabbit named Rosie.'
15:00:03 | EventBus             | DEBUG    | Published llm_speech_token from unknown
15:00:03 | ResponseParser       | DEBUG    | Sentence 4: 'She lived in a cozy burrow beneath a beautiful, old oak tree...'
15:00:03 | EventBus             | DEBUG    | Published llm_speech_token from unknown
15:00:03 | ResponseParser       | DEBUG    | Sentence 5: 'Rosie was always curious, and one day, she decided to explor...'
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.835
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
[AI] There was once a tiny, adventurous rabbit named Rosie.
[AI] She lived in a cozy burrow beneath a beautiful, old oak tree.
[AI] Rosie was always curious, and one day, she decided to explore the world above ground.
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.797
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.761
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.541
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | EventBus             | DEBUG    | Published pause_marker from unknown
15:00:03 | ResponseParser       | DEBUG    | Pause marker [...] at sentence 6
15:00:03 | EventBus             | DEBUG    | Published llm_speech_token from unknown
15:00:03 | ResponseParser       | DEBUG    | Sentence 6: 'She hopped up the tree trunk, feeling the warm sun on her fu...'
15:00:03 | EventBus             | DEBUG    | Published llm_stream_done from unknown
15:00:03 | EventBus             | DEBUG    | Published turn_complete from Session
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | AudioPlayer          | DEBUG    | Injected 300ms silence
[AI] She hopped up the tree trunk, feeling the warm sun on her fur, and discovered a hidden world of birds singing sweet melodies and leaves rustling in the gentle breeze.
15:00:03 | ShortTermMemory      | DEBUG    | Turn count: 2
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.389
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | groq._base_client    | DEBUG    | Request options: {'method': 'post', 'url': '/openai/v1/chat/completions', 'files': None, 'idempotency_key': 'stainless-python-retry-0460c9d7-7592-4c57-b09a-4e0d8a12fab0', 'json_data': {'messages': [{'role': 'user', 'content': 'Previous user message'}, {'role': 'assistant', 'content': "[cheerful] Oh, I'd love to! Let me think for a moment... [...] Alright, here's a little tale for you. There was once a tiny, adventurous rabbit named Rosie. She lived in a cozy burrow beneath a beautiful, old oak tree. Rosie was always curious, and one day, she decided to explore the world above ground. [...] She hopped up the tree trunk, feeling the warm sun on her fur, and discovered a hidden world of birds singing sweet melodies and leaves rustling in the gentle breeze."}], 'model': 'meta-llama/llama-guard-4-12b', 'max_tokens': 100, 'temperature': 0.0}}
15:00:03 | groq._base_client    | DEBUG    | Sending HTTP Request: POST https://api.groq.com/openai/v1/chat/completions
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.692
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.516
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.599
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.590
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.585
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.570
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.777
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.694
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.497
15:00:03 | MicStream            | DEBUG    | V3 filter_active = False
15:00:03 | MicStream            | DEBUG    | V3 filter_active = True
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.785
15:00:03 | AriaVoiceFilter      | DEBUG    | Aria embedding updated (64171 samples)
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.924 > 0.85 threshold
15:00:03 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:03 | EventBus             | DEBUG    | Published tts_sentence_done from TTSClient
15:00:03 | TTSClient            | INFO     | TTS: sentence 0 synthesized
15:00:03 | TTSClient            | INFO     | TTS: synthesizing sentence 1: 'Let me think for a moment...'
15:00:03 | phonemizer           | WARNING  | words count mismatch on 100.0% of the lines (1/1)
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
[TTS] chunk 14661 bytes
[TTS] chunk 13450 bytes
[TTS] chunk 13739 bytes
[TTS] chunk 14394 bytes
[TTS] chunk 13770 bytes
[TTS] chunk 14070 bytes
[TTS] chunk 13852 bytes
[TTS] chunk 13877 bytes
[TTS] chunk 14571 bytes
[TTS] chunk 13840 bytes
[TTS] chunk 13997 bytes
[TTS] chunk 14118 bytes
[TTS] chunk 15145 bytes
[TTS] chunk 16231 bytes
[TTS] chunk 234 bytes
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | groq._base_client    | DEBUG    | HTTP Response: POST https://api.groq.com/openai/v1/chat/completions "200 OK" Headers({'date': 'Sun, 01 Mar 2026 21:00:03 GMT', 'content-type': 'application/json', 'transfer-encoding': 'chunked', 'connection': 'keep-alive', 'cache-control': 'private, max-age=0, no-store, no-cache, must-revalidate', 'server': 'cloudflare', 'vary': 'Origin', 'x-groq-region': 'dal', 'x-ratelimit-limit-requests': '14400', 'x-ratelimit-limit-tokens': '15000', 'x-ratelimit-remaining-requests': '14397', 'x-ratelimit-remaining-tokens': '14413', 'x-ratelimit-reset-requests': '18s', 'x-ratelimit-reset-tokens': '2.348s', 'x-request-id': 'req_01kjnk6yk3ea2trtprm3yefhkc', 'cf-cache-status': 'DYNAMIC', 'strict-transport-security': 'max-age=15552000', 'content-encoding': 'gzip', 'cf-ray': '9d5b16ca18b5ddb1-DFW', 'alt-svc': 'h3=":443"; ma=86400'})
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.807
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.736
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.557
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.922 > 0.85 threshold
15:00:03 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.926 > 0.85 threshold
15:00:03 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:03 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.717
15:00:03 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:03 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.886 > 0.85 threshold
15:00:04 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.556
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.585
15:00:04 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.859 > 0.85 threshold
15:00:04 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.717
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.523
15:00:04 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.858 > 0.85 threshold
15:00:04 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:04 | MicStream            | DEBUG    | V3 filter_active = False
15:00:04 | MicStream            | DEBUG    | V3 filter_active = True
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.433
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.809
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_sentence_done from TTSClient
15:00:04 | TTSClient            | INFO     | TTS: sentence 1 synthesized
15:00:04 | TTSClient            | INFO     | TTS: synthesizing sentence 2: 'Alright, here's a little tale for you.'
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
[TTS] chunk 16112 bytes
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.706
[TTS] chunk 13752 bytes
[TTS] chunk 13871 bytes
[TTS] chunk 13839 bytes
[TTS] chunk 14207 bytes
[TTS] chunk 13764 bytes
[TTS] chunk 14078 bytes
[TTS] chunk 13679 bytes
[TTS] chunk 14034 bytes
[TTS] chunk 13809 bytes
[TTS] chunk 13879 bytes
[TTS] chunk 13956 bytes
[TTS] chunk 14083 bytes
[TTS] chunk 13956 bytes
[TTS] chunk 15226 bytes
[TTS] chunk 16241 bytes
[TTS] chunk 234 bytes
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.722
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.467
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.523
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.617
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.845
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.647
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.834
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.742
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.614
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.833
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.937 > 0.85 threshold
15:00:04 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.899 > 0.85 threshold
15:00:04 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.566
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.915 > 0.85 threshold
15:00:04 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.787
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.984 > 0.85 threshold
15:00:04 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.813
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:04 | EventBus             | DEBUG    | Published tts_sentence_done from TTSClient
15:00:04 | TTSClient            | INFO     | TTS: sentence 2 synthesized
15:00:04 | TTSClient            | INFO     | TTS: synthesizing sentence 3: 'There was once a tiny, adventurous rabbit named Rosie.'
15:00:04 | phonemizer           | WARNING  | words count mismatch on 100.0% of the lines (1/1)
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.804
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
[TTS] chunk 16255 bytes
[TTS] chunk 13890 bytes
[TTS] chunk 13544 bytes
[TTS] chunk 13618 bytes
[TTS] chunk 13858 bytes
[TTS] chunk 14063 bytes
[TTS] chunk 14364 bytes
[TTS] chunk 14173 bytes
[TTS] chunk 13791 bytes
[TTS] chunk 13820 bytes
[TTS] chunk 13936 bytes
[TTS] chunk 13914 bytes
[TTS] chunk 13913 bytes
[TTS] chunk 14096 bytes
[TTS] chunk 13886 bytes
[TTS] chunk 13861 bytes
[TTS] chunk 14147 bytes
[TTS] chunk 13807 bytes
[TTS] chunk 14014 bytes
[TTS] chunk 14089 bytes
[TTS] chunk 14050 bytes
[TTS] chunk 15900 bytes
[TTS] chunk 8334 bytes
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.920 > 0.85 threshold
15:00:04 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.849
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:04 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.945 > 0.85 threshold
15:00:04 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:04 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:04 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.745
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.905 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.839
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.665
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.758
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.833
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.830
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.898 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.735
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.755
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.849
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.793
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.854 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.876 > 0.85 threshold
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=234)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 234 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.896 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.946 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.926 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.746
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.935 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.915 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | MicStream            | DEBUG    | V3 filter_active = False
15:00:05 | MicStream            | DEBUG    | V3 filter_active = True
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.978 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.898 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.904 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:05 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:05 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:05 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:05 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:05 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:05 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:05 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:05 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:05 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:05 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient


TTS] chunk 14146 bytes
[TTS] chunk 15055 bytes
[TTS] chunk 8412 bytes
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.765
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.893 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.746
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.418
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.908 > 0.85 threshold
15:00:05 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.768
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:05 | InterruptionDetector | INFO     | Interrupt candidate: prob=0.992 rms=0.0538 gate=0.0469
15:00:05 | InterruptionDetector | DEBUG    | Interrupt timer accumulating (prob=0.992, rms=0.0538)
15:00:05 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.609
15:00:05 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:05 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.989 > 0.85 threshold
15:00:06 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.827
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.740
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.766
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.675
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.847
15:00:06 | AriaVoiceFilter      | DEBUG    | Aria embedding updated (15914 samples)
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.668
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.832
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.756
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.871 > 0.85 threshold
15:00:06 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.903 > 0.85 threshold
15:00:06 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.954 > 0.85 threshold
15:00:06 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.658
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.656
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.886 > 0.85 threshold
15:00:06 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.850 > 0.85 threshold
15:00:06 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.769
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.637
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.824
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.975 > 0.85 threshold
15:00:06 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.890 > 0.85 threshold
15:00:06 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:06 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:06 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:06 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:06 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:06 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:06 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:06 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:06 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:06 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:06 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:06 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:06 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
TTS] chunk 14276 bytes
[TTS] chunk 14779 bytes
[TTS] chunk 8400 bytes
15:00:07 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.487
15:00:07 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:07 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:07 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:07 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:07 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.815
15:00:07 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:07 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:07 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:07 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:07 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.544
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.469
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.815
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.720
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.764
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.822
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.681
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.801
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.568
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.709
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.857 > 0.85 threshold
15:00:08 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.832
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.572
15:00:08 | AriaVoiceFilter      | DEBUG    | Aria embedding updated (16000 samples)
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.554
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.627
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.889 > 0.85 threshold
15:00:08 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.709
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.934 > 0.85 threshold
15:00:08 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.937 > 0.85 threshold
15:00:08 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.801
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.761
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.989 > 0.85 threshold
15:00:08 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | InterruptionDetector | INFO     | Interrupt candidate: prob=1.000 rms=0.1774 gate=0.0469
15:00:08 | InterruptionDetector | DEBUG    | Interrupt timer accumulating (prob=1.000, rms=0.1774)
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 0ms (prob=1.00, ratio=9.5×) — waiting
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 1ms (prob=1.00, ratio=8.4×) — waiting
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 1ms (prob=1.00, ratio=6.3×) — waiting
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 2ms (prob=1.00, ratio=4.2×) — waiting
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 2ms (prob=1.00, ratio=5.2×) — waiting
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 2ms (prob=1.00, ratio=4.5×) — waiting
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 3ms (prob=1.00, ratio=3.1×) — waiting
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 3ms (prob=1.00, ratio=3.9×) — waiting
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.846
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.758
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 59ms (prob=1.00, ratio=8.6×) — waiting
15:00:08 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.850 > 0.85 threshold
15:00:08 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.765
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.809
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 141ms (prob=1.00, ratio=6.7×) — waiting
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 158ms (prob=1.00, ratio=5.8×) — waiting
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.770
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.796
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | InterruptionDetector | DEBUG    | Gate 1: 203ms (prob=1.00, ratio=5.1×) — waiting
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.868 > 0.85 threshold
15:00:08 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.873 > 0.85 threshold
15:00:08 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:08 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.894 > 0.85 threshold
15:00:08 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:08 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:08 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.925 > 0.85 threshold
15:00:09 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.947 > 0.85 threshold
15:00:09 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.876 > 0.85 threshold
15:00:09 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.835
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.737
15:00:09 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.730
15:00:09 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.710
15:00:09 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.794
15:00:09 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.678
15:00:09 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.806
15:00:09 | MicStream            | DEBUG    | V3 filter_active = False
15:00:09 | MicStream            | DEBUG    | V3 filter_active = True
15:00:09 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.843
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 PASSED: spectral=0.782
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.890 > 0.85 threshold
15:00:09 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.903 > 0.85 threshold
15:00:09 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.942 > 0.85 threshold
15:00:09 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 BLOCKED (spectral): sim=0.990 > 0.85 threshold
15:00:09 | MicStream            | DEBUG    | Gate0 BLOCKED echo chunk
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | Gate0                | DEBUG    | Gate0 EMA updated (bins=6, samples=320)
15:00:09 | AudioPlayer          | DEBUG    | TTS chunk: 320 samples → AriaFilter + Gate0 fed
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00:09 | EventBus             | DEBUG    | Published tts_chunk_ready from TTSClient
15:00