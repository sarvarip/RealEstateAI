import { FileSearch } from "lucide-react";
import { DocumentUpload } from "./DocumentUpload";

interface EmptyStateProps {
	onUpload: (files: File[]) => void;
	uploading?: boolean;
}

export function EmptyState({ onUpload, uploading }: EmptyStateProps) {
	return (
		<div className="flex flex-col items-center px-4">
			<div className="mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-neutral-900">
				<FileSearch className="h-7 w-7 text-white" />
			</div>
			<h2 className="mb-2 text-lg font-semibold text-neutral-800">
				Upload a document to get started
			</h2>
			<p className="mb-8 max-w-sm text-center text-sm text-neutral-500">
				Ask questions about leases, title reports, contracts, and other legal
				documents
			</p>
			<DocumentUpload onUpload={onUpload} uploading={uploading} />
		</div>
	);
}
