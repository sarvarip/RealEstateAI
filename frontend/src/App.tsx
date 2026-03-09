import { useCallback, useEffect, useState } from "react";
import { ChatSidebar } from "./components/ChatSidebar";
import { ChatWindow } from "./components/ChatWindow";
import { DocumentViewer } from "./components/DocumentViewer";
import { TooltipProvider } from "./components/ui/tooltip";
import { useConversations } from "./hooks/use-conversations";
import { useDocument } from "./hooks/use-document";
import { useMessages } from "./hooks/use-messages";
import type { Citation } from "./types";

export default function App() {
	const {
		conversations,
		selectedId,
		loading: conversationsLoading,
		create,
		select,
		remove,
		refresh: refreshConversations,
	} = useConversations();

	const {
		messages,
		loading: messagesLoading,
		error: messagesError,
		thinking,
		toolStatus,
		proposal,
		send,
		executeReport,
	} = useMessages(selectedId);

	const {
		document,
		documents,
		activeDocId,
		setActiveDocId,
		upload,
		refresh: refreshDocument,
	} = useDocument(selectedId);

	const [targetPage, setTargetPage] = useState<number | null>(null);
	const [highlightText, setHighlightText] = useState<string | null>(null);

	useEffect(() => {
		setTargetPage(null);
		setHighlightText(null);
	}, [selectedId]);

	const handleSend = useCallback(
		async (content: string) => {
			await send(content);
			refreshConversations();
		},
		[send, refreshConversations],
	);

	const handleUpload = useCallback(
		async (files: File[]) => {
			const doc = await upload(files);
			if (doc) {
				refreshDocument();
				refreshConversations();
			}
		},
		[upload, refreshDocument, refreshConversations],
	);

	const handleCreate = useCallback(async () => {
		await create();
	}, [create]);

	const handleCitationClick = useCallback(
		(citation: Citation) => {
			if (citation.document_id) {
				const matchedDoc = documents.find(
					(d) => d.id === citation.document_id,
				);
				if (matchedDoc) {
					setActiveDocId(matchedDoc.id);
				}
			}
			setTargetPage(citation.page);
			setHighlightText(citation.quote || null);
		},
		[documents, setActiveDocId],
	);

	const handleClearHighlight = useCallback(() => {
		setHighlightText(null);
		setTargetPage(null);
	}, []);

	return (
		<TooltipProvider delayDuration={200}>
			<div className="flex h-screen bg-neutral-50">
				<ChatSidebar
					conversations={conversations}
					selectedId={selectedId}
					loading={conversationsLoading}
					onSelect={select}
					onCreate={handleCreate}
					onDelete={remove}
					documents={documents}
					activeDocId={activeDocId}
					onSelectDocument={setActiveDocId}
				/>

		<ChatWindow
			messages={messages}
			loading={messagesLoading}
			error={messagesError}
			thinking={thinking}
			toolStatus={toolStatus}
			proposal={proposal}
			hasDocument={documents.length > 0}
			conversationId={selectedId}
			onSend={handleSend}
			onUpload={handleUpload}
			onCitationClick={handleCitationClick}
			onExecuteReport={executeReport}
		/>

				<DocumentViewer
					document={document}
					targetPage={targetPage}
					highlightText={highlightText}
					onClearHighlight={handleClearHighlight}
				/>
			</div>
		</TooltipProvider>
	);
}
