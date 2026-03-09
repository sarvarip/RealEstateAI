import { AlertCircle, Loader2, RefreshCw } from "lucide-react";
import { useEffect, useRef } from "react";
import type { Citation, Message, SectionsProposal } from "../types";
import { ChatInput } from "./ChatInput";
import { EmptyState } from "./EmptyState";
import { MessageBubble, ThinkingBubble } from "./MessageBubble";
import { SectionProposal } from "./SectionProposal";

interface ChatWindowProps {
	messages: Message[];
	loading: boolean;
	error: string | null;
	thinking: boolean;
	toolStatus?: string | null;
	proposal?: SectionsProposal | null;
	hasDocument: boolean;
	conversationId: string | null;
	onSend: (content: string) => void;
	onUpload: (files: File[]) => void;
	onCitationClick?: (citation: Citation) => void;
	onExecuteReport?: (sectionIds: string[]) => void;
}

export function ChatWindow({
	messages,
	loading,
	error,
	thinking,
	toolStatus,
	proposal,
	hasDocument,
	conversationId,
	onSend,
	onUpload,
	onCitationClick,
	onExecuteReport,
}: ChatWindowProps) {
	const scrollRef = useRef<HTMLDivElement>(null);

	const messagesLength = messages.length;
	// biome-ignore lint/correctness/useExhaustiveDependencies: messages and thinking are intentional triggers for auto-scroll
	useEffect(() => {
		if (scrollRef.current) {
			scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
		}
	}, [messagesLength, thinking]);

	if (!conversationId) {
		return (
			<div className="flex flex-1 items-center justify-center bg-neutral-50">
				<div className="text-center">
					<p className="text-sm text-neutral-400">
						Select a conversation or create a new one
					</p>
				</div>
			</div>
		);
	}

	if (loading) {
		return (
			<div className="flex flex-1 items-center justify-center bg-white">
				<Loader2 className="h-6 w-6 animate-spin text-neutral-400" />
			</div>
		);
	}

	if (messages.length === 0 && !thinking) {
		return (
			<div className="flex flex-1 flex-col bg-white">
				<div className="flex flex-1 items-center justify-center">
					{hasDocument ? (
						<div className="text-center">
							<p className="text-sm text-neutral-500">
								Document uploaded. Ask a question to get started.
							</p>
						</div>
					) : (
						<EmptyState onUpload={onUpload} />
					)}
				</div>
				<ChatInput
					onSend={onSend}
					onUpload={onUpload}
					disabled={thinking}
					hasDocument={hasDocument}
				/>
			</div>
		);
	}

	const lastMessage = messages[messages.length - 1];
	const needsRetry =
		!thinking && !loading && lastMessage?.role === "user";

	return (
		<div className="flex flex-1 flex-col bg-white">
			{error && (
				<div className="mx-4 mt-2 rounded-lg bg-red-50 px-4 py-2 text-sm text-red-600">
					{error}
				</div>
			)}

			<div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-4">
				<div className="mx-auto max-w-2xl space-y-1">
					{messages.map((message) => (
						<MessageBubble
							key={message.id}
							message={message}
							onCitationClick={onCitationClick}
							onRetry={
								message.role === "assistant" &&
								message.citations?.some((c) => !c.verified)
									? () => {
											const prev = messages
												.filter((m) => m.role === "user")
												.pop();
											if (prev) onSend(prev.content);
										}
									: undefined
							}
						/>
					))}
					{needsRetry && (
						<div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 mt-2">
							<AlertCircle className="h-4 w-4 flex-shrink-0 text-red-500" />
							<span className="text-sm text-red-700">
								No response received — the connection may have been interrupted.
							</span>
							<button
								type="button"
								onClick={() => onSend(lastMessage.content)}
								className="ml-auto flex items-center gap-1.5 rounded-md bg-red-100 px-3 py-1 text-xs font-medium text-red-700 hover:bg-red-200 transition-colors"
							>
								<RefreshCw className="h-3 w-3" />
								Retry
							</button>
						</div>
					)}
					{proposal && !thinking && (
						<SectionProposal
							proposal={proposal}
							onExecute={(ids) => onExecuteReport?.(ids)}
							disabled={thinking}
						/>
					)}
					{thinking && <ThinkingBubble toolStatus={toolStatus} />}
				</div>
			</div>

			<ChatInput
				onSend={onSend}
				onUpload={onUpload}
				disabled={thinking}
				hasDocument={hasDocument}
			/>
		</div>
	);
}
