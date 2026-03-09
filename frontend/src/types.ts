export interface Conversation {
	id: string;
	title: string;
	created_at: string;
	updated_at: string;
	has_document: boolean;
}

export interface Citation {
	index: number;
	document_id: string | null;
	filename: string;
	page: number;
	quote: string;
	verified: boolean;
}

export interface AnswerSegment {
	text: string;
	citations: Citation[];
}

export interface Message {
	id: string;
	conversation_id: string;
	role: "user" | "assistant" | "system";
	content: string;
	sources_cited: number;
	citations: Citation[];
	segments?: AnswerSegment[];
	proposed_sections?: ReportSection[];
	doc_summary?: string;
	created_at: string;
}

export interface Document {
	id: string;
	conversation_id: string;
	filename: string;
	page_count: number;
	uploaded_at: string;
}

export interface ReportSection {
	id: string;
	title: string;
	description: string;
}

export interface SectionsProposal {
	sections: ReportSection[];
	docSummary: string;
}

export interface ConversationDetail extends Conversation {
	document?: Document;
	documents: Document[];
}
