import { MessageSquarePlus, Trash2 } from "lucide-react";
import { useState } from "react";
import { relativeTime } from "../lib/utils";
import type { Conversation } from "../types";
import { Button } from "./ui/button";

interface ChatSidebarProps {
	conversations: Conversation[];
	selectedId: string | null;
	loading: boolean;
	onSelect: (id: string) => void;
	onCreate: () => void;
	onDelete: (id: string) => void;
}

export function ChatSidebar({
	conversations,
	selectedId,
	loading,
	onSelect,
	onCreate,
	onDelete,
}: ChatSidebarProps) {
	const [hoveredId, setHoveredId] = useState<string | null>(null);

	return (
		<div className="flex h-full w-[250px] flex-shrink-0 flex-col border-r border-neutral-200 bg-white">
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
	);
}
