import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../lib/api";
import type { AnswerSegment, Citation, Message } from "../types";

export function useMessages(conversationId: string | null) {
	const [messages, setMessages] = useState<Message[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [thinking, setThinking] = useState(false);
	const abortRef = useRef<AbortController | null>(null);

	const refresh = useCallback(async () => {
		if (!conversationId) {
			setMessages([]);
			return;
		}
		try {
			setLoading(true);
			setError(null);
			const data = await api.fetchMessages(conversationId);
			setMessages(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to load messages");
		} finally {
			setLoading(false);
		}
	}, [conversationId]);

	useEffect(() => {
		refresh();
		return () => {
			if (abortRef.current) {
				abortRef.current.abort();
			}
		};
	}, [refresh]);

	const send = useCallback(
		async (content: string) => {
			if (!conversationId || thinking) return;

			const userMessage: Message = {
				id: `temp-${Date.now()}`,
				conversation_id: conversationId,
				role: "user",
				content,
				sources_cited: 0,
				citations: [],
				created_at: new Date().toISOString(),
			};

			setMessages((prev) => [...prev, userMessage]);
			setThinking(true);
			setError(null);

			let pendingSegments: AnswerSegment[] | undefined;
			let pendingCitations: Citation[] = [];

			try {
				const response = await api.sendMessage(conversationId, content);

				if (!response.body) {
					throw new Error("No response body");
				}

				const reader = response.body.getReader();
				const decoder = new TextDecoder();
				let buffer = "";

				while (true) {
					const { done, value } = await reader.read();
					if (done) break;

					buffer += decoder.decode(value, { stream: true });
					const lines = buffer.split("\n");
					buffer = lines.pop() ?? "";

					for (const line of lines) {
						const trimmed = line.trim();
						if (!trimmed || !trimmed.startsWith("data: ")) continue;

						const data = trimmed.slice(6);
						if (data === "[DONE]") continue;

						try {
							const parsed = JSON.parse(data) as {
								type?: string;
								status?: string;
								segments?: AnswerSegment[];
								citations?: Citation[];
								message?: Message;
								content?: string;
							};

							if (parsed.type === "status") {
								// "thinking" status — keep showing loader
							} else if (parsed.type === "segments") {
								pendingSegments = parsed.segments;
								pendingCitations =
									parsed.segments?.flatMap(
										(s: AnswerSegment) => s.citations,
									) ?? [];
							} else if (parsed.type === "message" && parsed.message) {
								const msg: Message = {
									...parsed.message,
									citations:
										parsed.message.citations ?? pendingCitations,
									segments:
										parsed.message.segments ?? pendingSegments,
								};
								setMessages((prev) => [...prev, msg]);
							} else if (parsed.type === "content" && parsed.content) {
								// Error fallback — plain content with no structured data
								const fallbackMsg: Message = {
									id: `err-${Date.now()}`,
									conversation_id: conversationId,
									role: "assistant",
									content: parsed.content,
									sources_cited: 0,
									citations: [],
									created_at: new Date().toISOString(),
								};
								setMessages((prev) => [...prev, fallbackMsg]);
							}
						} catch {
							// Skip invalid JSON
						}
					}
				}

				const freshMessages = await api.fetchMessages(conversationId);
				setMessages(freshMessages);
			} catch (err) {
				if (err instanceof DOMException && err.name === "AbortError") return;
				setError(err instanceof Error ? err.message : "Failed to send message");
			} finally {
				setThinking(false);
			}
		},
		[conversationId, thinking],
	);

	return {
		messages,
		loading,
		error,
		thinking,
		send,
		refresh,
	};
}
