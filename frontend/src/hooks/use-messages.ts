import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../lib/api";
import type {
	AnswerSegment,
	Citation,
	Message,
	ReportSection,
	SectionsProposal,
} from "../types";

export function useMessages(conversationId: string | null) {
	const [messages, setMessages] = useState<Message[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [thinking, setThinking] = useState(false);
	const [toolStatus, setToolStatus] = useState<string | null>(null);
	const [proposal, setProposal] = useState<SectionsProposal | null>(null);
	const abortRef = useRef<AbortController | null>(null);
	const statusTimeoutRef = useRef<number>(0);

	const refresh = useCallback(async () => {
		if (!conversationId) {
			setMessages([]);
			setProposal(null);
			return;
		}
		try {
			setLoading(true);
			setError(null);
			const data = await api.fetchMessages(conversationId);
			setMessages(data);

			// Restore the proposal widget only if the LAST assistant message
			// contains proposed_sections. If Phase 2 was already executed,
			// the last assistant message will be the report (no sections),
			// so the widget correctly stays hidden.
			const lastAssistant = [...data]
				.reverse()
				.find((m) => m.role === "assistant");
			if (
				lastAssistant?.proposed_sections &&
				lastAssistant.proposed_sections.length > 0
			) {
				setProposal({
					sections: lastAssistant.proposed_sections,
					docSummary: lastAssistant.doc_summary ?? "",
				});
			} else {
				setProposal(null);
			}
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

	const processStream = useCallback(
		async (
			response: Response,
			opts?: { isReportExecution?: boolean },
		) => {
			if (!response.body || !conversationId) return;

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = "";
			let pendingSegments: AnswerSegment[] | undefined;
			let pendingCitations: Citation[] = [];
			let gotResponse = false;

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
						const parsed = JSON.parse(data) as Record<string, unknown>;

						if (parsed.type === "status") {
							const s = parsed.status as string;
							const phase = parsed.phase as
								| string
								| undefined;
							const section = parsed.section as
								| string
								| undefined;

							const phaseLabel =
								phase === "summary"
									? "Analysing"
									: phase === "planning"
										? "Planning"
										: section
											? section
											: null;
							const prefix = phaseLabel
								? `${phaseLabel}: `
								: "";

							if (s === "searching") {
								setToolStatus(
									`${prefix}Searching: ${(parsed.query as string) ?? ""}`.trim(),
								);
							} else if (s === "reading") {
								setToolStatus(
									`${prefix}Reading page ${parsed.page} of ${(parsed.filename as string) ?? "document"}`,
								);
							} else if (s === "summarising") {
								setToolStatus("Analysing documents…");
							} else if (s === "planning") {
								setToolStatus("Planning report sections…");
							} else {
								setToolStatus(null);
							}
							clearTimeout(statusTimeoutRef.current);
							const timeout =
								phase === "summary" || phase === "planning"
									? 30000
									: 8000;
							statusTimeoutRef.current = window.setTimeout(
								() => {
									setToolStatus("Generating response…");
								},
								timeout,
							);
						} else if (parsed.type === "segments") {
							pendingSegments = parsed.segments as AnswerSegment[];
							pendingCitations =
								(parsed.segments as AnswerSegment[])?.flatMap(
									(seg: AnswerSegment) => seg.citations,
								) ?? [];
					} else if (parsed.type === "sections_proposal") {
						const sections =
							(parsed.sections as ReportSection[]) ?? [];
						const docSummary =
							(parsed.doc_summary as string) ?? "";
						if (sections.length > 0) {
							setProposal({ sections, docSummary });
						}
						} else if (parsed.type === "message" && parsed.message) {
							gotResponse = true;
							const msg = parsed.message as Message;
							const fullMsg: Message = {
								...msg,
								citations: msg.citations ?? pendingCitations,
								segments: msg.segments ?? pendingSegments,
							};
							setMessages((prev) => [...prev, fullMsg]);
						} else if (parsed.type === "content" && parsed.content) {
							gotResponse = true;
							const fallbackMsg: Message = {
								id: `err-${Date.now()}`,
								conversation_id: conversationId,
								role: "assistant",
								content: parsed.content as string,
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

			if (!gotResponse) {
				setError(
					"Connection lost — the server may have restarted. Please try again.",
				);
			}

			const freshMessages = await api.fetchMessages(conversationId);
			setMessages(freshMessages);
		},
		[conversationId],
	);

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
			setToolStatus(null);
			setError(null);
			setProposal(null);

			try {
				const response = await api.sendMessage(conversationId, content);
				await processStream(response);
			} catch (err) {
				if (err instanceof DOMException && err.name === "AbortError") return;
				setError(
					err instanceof Error ? err.message : "Failed to send message",
				);
			} finally {
				clearTimeout(statusTimeoutRef.current);
				setThinking(false);
				setToolStatus(null);
			}
		},
		[conversationId, thinking, processStream],
	);

	const executeReport = useCallback(
		async (sectionIds: string[]) => {
			if (!conversationId || thinking || !proposal) return;

			const selectedSections = proposal.sections.filter((s) =>
				sectionIds.includes(s.id),
			);
			if (selectedSections.length === 0) return;

			// Build a descriptive message showing which sections were selected,
			// so the conversation flow is clear (instead of repeating the original prompt)
			const sectionList = selectedSections.map((s) => s.title).join(", ");
			const content = `Generate report sections: ${sectionList}`;

			// Show the selected sections as a user message in the UI
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
			setToolStatus("Generating report sections…");
			setError(null);
			setProposal(null);

			try {
				const response = await api.sendMessage(conversationId, content, {
					reportSections: selectedSections,
					docSummary: proposal.docSummary,
				});
				await processStream(response, { isReportExecution: true });
			} catch (err) {
				if (err instanceof DOMException && err.name === "AbortError") return;
				setError(
					err instanceof Error ? err.message : "Failed to generate report",
				);
			} finally {
				clearTimeout(statusTimeoutRef.current);
				setThinking(false);
				setToolStatus(null);
			}
		},
		[conversationId, thinking, proposal, messages, processStream],
	);

	return {
		messages,
		loading,
		error,
		thinking,
		toolStatus,
		proposal,
		send,
		executeReport,
		refresh,
	};
}
