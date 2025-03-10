package ai.mlc.mlcllm

import ai.mlc.mlcllm.OpenAIProtocol.*
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ReceiveChannel
import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlin.concurrent.thread
import java.util.UUID
import java.util.logging.Logger

class BackgroundWorker(private val task: () -> Unit) {

    fun start() {
        thread(start = true) {
            task()
        }
    }
}

class MLCEngine {

    private val state: EngineState
    private val jsonFFIEngine: JSONFFIEngine
    val chat: Chat
    private val threads = mutableListOf<BackgroundWorker>()

    init {
        state = EngineState()
        jsonFFIEngine = JSONFFIEngine()
        chat = Chat(jsonFFIEngine, state)

        jsonFFIEngine.initBackgroundEngine { result ->
            state.streamCallback(result)
        }

        val backgroundWorker = BackgroundWorker {
            Thread.currentThread().priority = Thread.MAX_PRIORITY
            jsonFFIEngine.runBackgroundLoop()
        }

        val backgroundStreamBackWorker = BackgroundWorker {
            jsonFFIEngine.runBackgroundStreamBackLoop()
        }

        threads.add(backgroundWorker)
        threads.add(backgroundStreamBackWorker)

        backgroundWorker.start()
        backgroundStreamBackWorker.start()
    }

    fun reload(modelPath: String, modelLib: String) {
        val engineConfig = """
            {
                "model": "$modelPath",
                "model_lib": "system://$modelLib",
                "mode": "interactive"
            }
        """
        jsonFFIEngine.reload(engineConfig)
    }

    fun reset() {
        jsonFFIEngine.reset()
    }

    fun unload() {
        jsonFFIEngine.unload()
    }
}

data class RequestState(
    val request: ChatCompletionRequest,
    val continuation: Channel<ChatCompletionStreamResponse>
)

class EngineState {

    private val logger = Logger.getLogger(EngineState::class.java.name)
    private val requestStateMap = mutableMapOf<String, RequestState>()

    suspend fun chatCompletion(
        jsonFFIEngine: JSONFFIEngine,
        request: ChatCompletionRequest
    ): ReceiveChannel<ChatCompletionStreamResponse> {
        val json = Json { encodeDefaults = true }
        val jsonRequest = json.encodeToString(request)
        val requestID = UUID.randomUUID().toString()
        val channel = Channel<ChatCompletionStreamResponse>(Channel.UNLIMITED)

        requestStateMap[requestID] = RequestState(request, channel)

        jsonFFIEngine.chatCompletion(jsonRequest, requestID)

        return channel
    }

    fun streamCallback(result: String?) {
        val json = Json { ignoreUnknownKeys = true }
        try {
            val responses: List<ChatCompletionStreamResponse> = json.decodeFromString(result ?: return)

            responses.forEach { res ->
                val requestState = requestStateMap[res.id] ?: return@forEach
                GlobalScope.launch {

                    res.usage?.let { finalUsage ->
                        requestState.request.stream_options?.let { streamOptions ->
                            if (streamOptions.include_usage) { // Ensure include_usage is used here
                                requestState.continuation.send(res)
                            }
                        }
                        requestState.continuation.close()
                        requestStateMap.remove(res.id)
                    } ?: run {
                        val sendResult = requestState.continuation.trySend(res)
                        if (sendResult.isFailure) {
                            // Handle the failure case if needed
                            logger.severe("Failed to send the response: ${sendResult.exceptionOrNull()}")
                        }
                    }
                }
            }
        } catch (e: Exception) {
            logger.severe("Kotlin JSON parsing error: $e, jsonsrc=$result")
        }
    }
}

class Chat(
    private val jsonFFIEngine: JSONFFIEngine,
    private val state: EngineState
) {
    val completions = Completions(jsonFFIEngine, state)
}

class Completions(
    private val jsonFFIEngine: JSONFFIEngine,
    private val state: EngineState
) {

    private val coroutineScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    suspend fun create(request: ChatCompletionRequest): ReceiveChannel<ChatCompletionStreamResponse> {
        return state.chatCompletion(jsonFFIEngine, request)
    }

    suspend fun create(
        messages: List<ChatCompletionMessage>,
        model: String? = null,
        frequency_penalty: Float? = null,
        presence_penalty: Float? = null,
        logprobs: Boolean = false,
        top_logprobs: Int = 0,
        logit_bias: Map<Int, Float>? = null,
        max_tokens: Int? = null,
        n: Int = 1,
        seed: Int? = null,
        stop: List<String>? = null,
        stream: Boolean = true,
        stream_options: StreamOptions? = null,
        temperature: Float? = null,
        top_p: Float? = null,
        tools: List<ChatTool>? = null,
        user: String? = null,
        response_format: ResponseFormat? = null,
        useChunking: Boolean = true, // Add the boolean parameter
        chunkSize: Int = 10 // Add the chunk size parameter
    ): ReceiveChannel<ChatCompletionStreamResponse> {
        if (!stream) {
            throw IllegalArgumentException("Only stream=true is supported in MLCKotlin")
        }

        val channel = Channel<ChatCompletionStreamResponse>(Channel.UNLIMITED)

        coroutineScope.launch {
            try {
                val aggregatedJson = mutableMapOf<String, MutableList<JsonElement>>()
                var previousResponseContent = ""

                if (useChunking) {
                    val chunks = messages.chunked(chunkSize)

                    for (chunk in chunks) {
                        val combinedMessages = if (previousResponseContent.isNotEmpty()) {
                            chunk + ChatCompletionMessage("system", previousResponseContent)
                        } else {
                            chunk
                        }

                        val request = ChatCompletionRequest(
                            messages = combinedMessages,
                            model = model,
                            frequency_penalty = frequency_penalty,
                            presence_penalty = presence_penalty,
                            logprobs = logprobs,
                            top_logprobs = top_logprobs,
                            logit_bias = logit_bias,
                            max_tokens = max_tokens,
                            n = n,
                            seed = seed,
                            stop = stop,
                            stream = stream,
                            stream_options = stream_options,
                            temperature = temperature,
                            top_p = top_p,
                            tools = tools,
                            user = user,
                            response_format = response_format
                        )
                        val chunkChannel = create(request)
                        for (response in chunkChannel) {
                            try {
                                val jsonResponse = Json.decodeFromString<JsonObject>(response.choices.first().message.content)
                                jsonResponse.forEach { (key, value) ->
                                    aggregatedJson.computeIfAbsent(key) { mutableListOf() }.addAll(value.jsonArray)
                                }
                                previousResponseContent = response.choices.first().message.content
                            } catch (e: Exception) {
                                Logger.getLogger(Completions::class.java.name).severe("Invalid JSON response: ${response.choices.first().message.content}")
                            }
                        }
                    }
                } else {
                    val request = ChatCompletionRequest(
                        messages = messages,
                        model = model,
                        frequency_penalty = frequency_penalty,
                        presence_penalty = presence_penalty,
                        logprobs = logprobs,
                        top_logprobs = top_logprobs,
                        logit_bias = logit_bias,
                        max_tokens = max_tokens,
                        n = n,
                        seed = seed,
                        stop = stop,
                        stream = stream,
                        stream_options = stream_options,
                        temperature = temperature,
                        top_p = top_p,
                        tools = tools,
                        user = user,
                        response_format = response_format
                    )
                    val responseChannel = create(request)
                    for (response in responseChannel) {
                        try {
                            val jsonResponse = Json.decodeFromString<JsonObject>(response.choices.first().message.content)
                            jsonResponse.forEach { (key, value) ->
                                aggregatedJson.computeIfAbsent(key) { mutableListOf() }.addAll(value.jsonArray)
                            }
                        } catch (e: Exception) {
                            Logger.getLogger(Completions::class.java.name).severe("Invalid JSON response: ${response.choices.first().message.content}")
                        }
                    }
                }

                // Create the aggregated JSON response
                val aggregatedResponse = JsonObject(
                    aggregatedJson.mapValues { JsonArray(it.value) }
                )

                // Send the aggregated response
                channel.send(
                    ChatCompletionStreamResponse(
                        id = UUID.randomUUID().toString(),
                        `object` = "chat.completion",
                        created = System.currentTimeMillis() / 1000,
                        model = model ?: "unknown",
                        choices = listOf(
                            ChatCompletionChoice(
                                index = 0,
                                message = ChatCompletionMessage(
                                    role = "assistant",
                                    content = aggregatedResponse.toString()
                                ),
                                finish_reason = "stop"
                            )
                        )
                    )
                )

            } catch (e: Exception) {
                // Handle any exceptions that occur during processing
                Logger.getLogger(Completions::class.java.name).severe("Error in create function: $e")
            } finally {
                channel.close()
            }
        }

        return channel
    }
}