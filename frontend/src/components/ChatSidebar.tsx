import { FileText, MessageSquarePlus, Trash2 } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { relativeTime } from "../lib/utils";
import type { Conversation, Document } from "../types";
import { Button } from "./ui/button";

interface ChatSidebarProps {
	conversations: Conversation[];
	selectedId: string | null;
	loading: boolean;
	onSelect: (id: string) => void;
	onCreate: () => void;
	onDelete: (id: string) => void;
	documents: Document[];
	activeDocId: string | null;
	onSelectDocument: (id: string) => void;
}

const DEFAULT_CHAT_PERCENT = 60;
const MIN_PERCENT = 20;
const MAX_PERCENT = 80;

export function ChatSidebar({
	conversations,
	selectedId,
	loading,
	onSelect,
	onCreate,
	onDelete,
	documents,
	activeDocId,
	onSelectDocument,
}: ChatSidebarProps) {
	const [hoveredId, setHoveredId] = useState<string | null>(null);
	const [chatPercent, setChatPercent] = useState(DEFAULT_CHAT_PERCENT);
	const containerRef = useRef<HTMLDivElement>(null);
	const dragging = useRef(false);

	const onDragStart = useCallback((e: React.MouseEvent) => {
		e.preventDefault();
		dragging.current = true;

		const onMove = (ev: MouseEvent) => {
			if (!dragging.current || !containerRef.current) return;
			const rect = containerRef.current.getBoundingClientRect();
			const pct = ((ev.clientY - rect.top) / rect.height) * 100;
			setChatPercent(Math.min(MAX_PERCENT, Math.max(MIN_PERCENT, pct)));
		};
		const onUp = () => {
			dragging.current = false;
			document.removeEventListener("mousemove", onMove);
			document.removeEventListener("mouseup", onUp);
			document.body.style.cursor = "";
			document.body.style.userSelect = "";
		};

		document.addEventListener("mousemove", onMove);
		document.addEventListener("mouseup", onUp);
		document.body.style.cursor = "row-resize";
		document.body.style.userSelect = "none";
	}, []);

	return (
		<div
			ref={containerRef}
			className="flex h-full w-[clamp(180px,16vw,280px)] flex-shrink-0 flex-col border-r border-neutral-200 bg-white"
		>
			{/* ── Chats section ── */}
			<div className="flex min-h-0 flex-col" style={{ height: `${chatPercent}%` }}>
				<div className="flex items-center justify-between border-b border-neutral-100 p-3">
					<span className="text-sm font-semibold text-neutral-700">Chats</span>
					<Button variant="ghost" size="icon" onClick={onCreate} title="New chat">
						<MessageSquarePlus className="h-4 w-4" />
					</Button>
				</div>

				<div className="flex-1 overflow-y-auto">
					<div className="p-2">
						{loading && conversations.length === 0 && (
							<div className="space-y-2 p-2">
								{[1, 2, 3].map((i) => (
									<div key={i} className="animate-pulse space-y-1">
										<div className="h-4 w-3/4 rounded bg-neutral-100" />
										<div className="h-3 w-1/2 rounded bg-neutral-50" />
									</div>
								))}
							</div>
						)}

						{!loading && conversations.length === 0 && (
							<p className="px-2 py-8 text-center text-xs text-neutral-400">
								No conversations yet
							</p>
						)}

						{conversations.map((conversation) => (
							<div key={conversation.id}>
								<button
									type="button"
									className={`group flex w-full items-center rounded-lg px-3 py-2.5 text-left transition-colors ${
										selectedId === conversation.id
											? "bg-neutral-100"
											: "hover:bg-neutral-50"
									}`}
									onClick={() => onSelect(conversation.id)}
									onMouseEnter={() => setHoveredId(conversation.id)}
									onMouseLeave={() => setHoveredId(null)}
								>
									<div className="min-w-0 flex-1 overflow-hidden">
										<p className="truncate text-sm font-medium text-neutral-800">
											{conversation.title}
										</p>
										<p className="mt-0.5 text-xs text-neutral-400">
											{relativeTime(conversation.updated_at)}
										</p>
									</div>

									<div className="ml-2 w-6 flex-shrink-0">
										{hoveredId === conversation.id && (
											<button
												type="button"
												className="rounded p-1 text-neutral-400 hover:bg-neutral-200 hover:text-red-500"
												onClick={(e) => {
													e.stopPropagation();
													onDelete(conversation.id);
												}}
												title="Delete conversation"
											>
												<Trash2 className="h-3.5 w-3.5" />
											</button>
										)}
									</div>
								</button>
							</div>
						))}
					</div>
				</div>
			</div>

			{/* ── Drag handle ── */}
			<div
				role="separator"
				aria-orientation="horizontal"
				onMouseDown={onDragStart}
				className="group flex h-1.5 flex-shrink-0 cursor-row-resize items-center justify-center hover:bg-neutral-100"
			>
				<div className="h-px w-8 rounded-full bg-neutral-200 transition-colors group-hover:bg-neutral-400" />
			</div>

			{/* ── Documents section ── */}
			<div className="flex min-h-0 flex-1 flex-col">
				<div className="flex items-center gap-1.5 border-b border-neutral-100 px-3 py-2.5">
					<FileText className="h-3.5 w-3.5 text-neutral-400" />
					<span className="text-xs font-semibold text-neutral-700">
						Documents
					</span>
					{documents.length > 0 && (
						<span className="ml-auto text-[10px] tabular-nums text-neutral-400">
							{documents.length}
						</span>
					)}
				</div>

				<div className="flex-1 overflow-y-auto">
					<div className="p-1.5">
						{documents.length === 0 ? (
							<p className="px-2 py-4 text-center text-[11px] text-neutral-400">
								No documents
							</p>
						) : (
							documents.map((doc) => (
								<button
									key={doc.id}
									type="button"
									onClick={() => onSelectDocument(doc.id)}
									className={`flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left transition-colors ${
										activeDocId === doc.id
											? "bg-neutral-100"
											: "hover:bg-neutral-50"
									}`}
									title={doc.filename}
								>
									<FileText
										className={`h-3.5 w-3.5 flex-shrink-0 ${
											activeDocId === doc.id
												? "text-neutral-600"
												: "text-neutral-300"
										}`}
									/>
									<div className="min-w-0 flex-1">
										<p
											className={`truncate text-xs ${
												activeDocId === doc.id
													? "font-semibold text-neutral-800"
													: "text-neutral-600"
											}`}
										>
											{doc.filename.replace(/\.pdf$/i, "")}
										</p>
										<p className="text-[10px] text-neutral-400">
											{doc.page_count} page
											{doc.page_count !== 1 ? "s" : ""}
										</p>
									</div>
								</button>
							))
						)}
					</div>
				</div>
			</div>
		</div>
	);
}
