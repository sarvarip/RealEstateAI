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
	const [pendingConvs, setPendingConvs] = useState<Set<string>>(new Set());
	const pendingConvsRef = useRef(pendingConvs);
	pendingConvsRef.current = pendingConvs;

	const statusTimeoutRef = useRef<number>(0);
	const convIdRef = useRef(conversationId);
	convIdRef.current = conversationId;

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

			// Don't restore the proposal widget if this conversation has a
			// background stream running — Phase 2 already cleared it and the
			// DB still has the Phase 1 proposal as the last assistant message.
			if (!pendingConvsRef.current.has(conversationId)) {
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
			}
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to load messages");
		} finally {
			setLoading(false);
		}
	}, [conversationId]);

	useEffect(() => {
		setThinking(false);
		setToolStatus(null);
		setError(null);
		clearTimeout(statusTimeoutRef.current);
		refresh();
	}, [refresh]);

	const processStream = useCallback(
		async (response: Response, ownerConvId: string, isReportExecution: boolean) => {
			if (!response.body) return;

			const isCurrent = () => convIdRef.current === ownerConvId;

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = "";
			let pendingSegments: AnswerSegment[] | undefined;
			let pendingCitations: Citation[] = [];
			let gotResponse = false;

			try {
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

							if (!isCurrent()) continue;

							if (parsed.type === "status") {
								const s = parsed.status as string;
								const phase = parsed.phase as string | undefined;
								const section = parsed.section as string | undefined;

								const phaseLabel =
									phase === "summary"
										? "Analysing"
										: phase === "planning"
											? "Planning"
											: section
												? section
												: null;
								const prefix = phaseLabel ? `${phaseLabel}: ` : "";

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
								statusTimeoutRef.current = window.setTimeout(() => {
									if (isCurrent()) {
										setToolStatus("Generating response…");
									}
								}, timeout);
							} else if (parsed.type === "segments") {
								pendingSegments = parsed.segments as AnswerSegment[];
								pendingCitations =
									(parsed.segments as AnswerSegment[])?.flatMap(
										(seg: AnswerSegment) => seg.citations,
									) ?? [];
							} else if (parsed.type === "sections_proposal") {
								const sections = (parsed.sections as ReportSection[]) ?? [];
								const docSummary = (parsed.doc_summary as string) ?? "";
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
									conversation_id: ownerConvId,
									role: "assistant",
									content: parsed.content as string,
									sources_cited: 0,
									citations: [],
									created_at: new Date().toISOString(),
								};
								setMessages((prev) => [...prev, fallbackMsg]);
							}
						} catch {
							// skip invalid JSON lines
						}
					}
				}
			} catch {
				// Stream error (network drop, etc). If user switched away, ignore.
				if (!isCurrent()) return;
			}

			// Stream ended. Remove from pending set.
			setPendingConvs((prev) => {
				const next = new Set(prev);
				next.delete(ownerConvId);
				return next;
			});

			clearTimeout(statusTimeoutRef.current);

			if (isCurrent()) {
				setThinking(false);
				setToolStatus(null);

				if (!gotResponse) {
					setError(
						"Connection lost — the server may have restarted. Please try again.",
					);
				}

				if (isReportExecution) {
					setProposal(null);
				}
			}

			// Always try to refresh from DB — the backend saved the message.
			// If user navigated back by the time this resolves, they see it.
			try {
				const freshMessages = await api.fetchMessages(ownerConvId);
				if (convIdRef.current === ownerConvId) {
					setMessages(freshMessages);
				}
			} catch {
				// network error on refresh — user can manually reload
			}
		},
		[],
	);

	const send = useCallback(
		async (content: string) => {
			if (!conversationId || thinking) return;

			const targetConvId = conversationId;

			const userMessage: Message = {
				id: `temp-${Date.now()}`,
				conversation_id: targetConvId,
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
			setPendingConvs((prev) => new Set(prev).add(targetConvId));

			try {
				const response = await api.sendMessage(targetConvId, content);
				await processStream(response, targetConvId, false);
			} catch (err) {
				if (convIdRef.current !== targetConvId) return;
				setError(
					err instanceof Error ? err.message : "Failed to send message",
				);
				setPendingConvs((prev) => {
					const next = new Set(prev);
					next.delete(targetConvId);
					return next;
				});
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

			const targetConvId = conversationId;

			const sectionList = selectedSections.map((s) => s.title).join(", ");
			const content = `Generate report sections: ${sectionList}`;

			const userMessage: Message = {
				id: `temp-${Date.now()}`,
				conversation_id: targetConvId,
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
			setPendingConvs((prev) => new Set(prev).add(targetConvId));

			try {
				const response = await api.sendMessage(targetConvId, content, {
					reportSections: selectedSections,
					docSummary: proposal.docSummary,
				});
				await processStream(response, targetConvId, true);
			} catch (err) {
				if (convIdRef.current !== targetConvId) return;
				setError(
					err instanceof Error ? err.message : "Failed to generate report",
				);
				setPendingConvs((prev) => {
					const next = new Set(prev);
					next.delete(targetConvId);
					return next;
				});
				setThinking(false);
				setToolStatus(null);
			}
		},
		[conversationId, thinking, proposal, messages, processStream],
	);

	const awaitingResponse = conversationId
		? pendingConvs.has(conversationId)
		: false;

	return {
		messages,
		loading,
		error,
		thinking,
		toolStatus,
		proposal,
		awaitingResponse,
		send,
		executeReport,
		refresh,
	};
}
