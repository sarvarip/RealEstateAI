import {
	ChevronLeft,
	ChevronRight,
	FileText,
	Loader2,
	X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Document as PDFDocument, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import { getDocumentUrl } from "../lib/api";
import type { Document } from "../types";
import { Button } from "./ui/button";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
	"pdfjs-dist/build/pdf.worker.min.mjs",
	import.meta.url,
).toString();

const MIN_WIDTH = 280;
const MAX_WIDTH = 700;
const DEFAULT_WIDTH = 400;

function normalizeForMatch(text: string): string {
	return text
		.toLowerCase()
		.normalize("NFKD")
		.replace(/[^\w\s]/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

interface DocumentViewerProps {
	document: Document | null;
	documents: Document[];
	activeDocId: string | null;
	onSelectDocument: (id: string) => void;
	targetPage: number | null;
	highlightText: string | null;
	onClearHighlight?: () => void;
}

export function DocumentViewer({
	document,
	documents,
	activeDocId,
	onSelectDocument,
	targetPage,
	highlightText,
	onClearHighlight,
}: DocumentViewerProps) {
	const [numPages, setNumPages] = useState<number>(0);
	const [currentPage, setCurrentPage] = useState(1);
	const [pdfLoading, setPdfLoading] = useState(true);
	const [pdfError, setPdfError] = useState<string | null>(null);
	const [width, setWidth] = useState(DEFAULT_WIDTH);
	const [dragging, setDragging] = useState(false);
	const containerRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		if (targetPage !== null && targetPage >= 1 && targetPage <= numPages) {
			setCurrentPage(targetPage);
		}
	}, [targetPage, numPages]);

	const applyHighlightToTextLayer = useCallback(() => {
		if (!containerRef.current) return;

		const textLayer = containerRef.current.querySelector(".textLayer");
		if (!textLayer) return;

		textLayer.querySelectorAll(".citation-highlight").forEach((el) => {
			(el as HTMLElement).style.backgroundColor = "";
			(el as HTMLElement).style.borderRadius = "";
			el.classList.remove("citation-highlight");
		});

		if (!highlightText) return;

		const spans = Array.from(
			textLayer.querySelectorAll('span[role="presentation"]'),
		) as HTMLElement[];
		if (spans.length === 0) return;

		const texts = spans.map((s) => s.textContent || "");
		const norms = texts.map(normalizeForMatch);

		const stripWs = (s: string) => s.replace(/\s+/g, "");

		const spanRanges: { start: number; end: number }[] = [];
		let pos = 0;
		for (const norm of norms) {
			const stripped = stripWs(norm);
			spanRanges.push({ start: pos, end: pos + stripped.length });
			pos += stripped.length;
		}

		const fullStripped = norms.map(stripWs).join("");
		const quoteStripped = stripWs(normalizeForMatch(highlightText));

		if (quoteStripped.length < 5) return;

		const matchIdx = fullStripped.indexOf(quoteStripped);
		if (matchIdx === -1) return;

		const matchEnd = matchIdx + quoteStripped.length;
		for (let i = 0; i < spanRanges.length; i++) {
			if (spanRanges[i].end > matchIdx && spanRanges[i].start < matchEnd) {
				spans[i].style.backgroundColor = "rgba(254, 240, 138, 0.6)";
				spans[i].style.borderRadius = "2px";
				spans[i].classList.add("citation-highlight");
			}
		}

		const first = textLayer.querySelector(".citation-highlight");
		if (first) {
			first.scrollIntoView({ block: "center", behavior: "smooth" });
		}
	}, [highlightText]);

	useEffect(() => {
		applyHighlightToTextLayer();
	}, [applyHighlightToTextLayer]);

	const handleMouseDown = useCallback(
		(e: React.MouseEvent) => {
			e.preventDefault();
			setDragging(true);

			const startX = e.clientX;
			const startWidth = width;

			const handleMouseMove = (moveEvent: MouseEvent) => {
				const delta = startX - moveEvent.clientX;
				const newWidth = Math.min(
					MAX_WIDTH,
					Math.max(MIN_WIDTH, startWidth + delta),
				);
				setWidth(newWidth);
			};

			const handleMouseUp = () => {
				setDragging(false);
				window.removeEventListener("mousemove", handleMouseMove);
				window.removeEventListener("mouseup", handleMouseUp);
			};

			window.addEventListener("mousemove", handleMouseMove);
			window.addEventListener("mouseup", handleMouseUp);
		},
		[width],
	);

	const handleDocSwitch = useCallback(
		(docId: string) => {
			onSelectDocument(docId);
			setCurrentPage(1);
			setPdfLoading(true);
			setPdfError(null);
		},
		[onSelectDocument],
	);

	const pdfPageWidth = width - 48;

	if (documents.length === 0) {
		return (
			<div
				style={{ width }}
				className="flex h-full flex-shrink-0 flex-col items-center justify-center border-l border-neutral-200 bg-neutral-50"
			>
				<FileText className="mb-3 h-10 w-10 text-neutral-300" />
				<p className="text-sm text-neutral-400">No document uploaded</p>
			</div>
		);
	}

	const pdfUrl = document ? getDocumentUrl(document.id) : null;

	return (
		<div
			ref={containerRef}
			style={{ width }}
			className="relative flex h-full flex-shrink-0 flex-col border-l border-neutral-200 bg-white"
		>
			{/* Resize handle */}
			<div
				className={`absolute top-0 left-0 z-10 h-full w-1.5 cursor-col-resize transition-colors hover:bg-neutral-300 ${
					dragging ? "bg-neutral-400" : ""
				}`}
				onMouseDown={handleMouseDown}
			/>

			{/* Document tabs */}
			{documents.length > 1 && (
				<div className="flex gap-0 overflow-x-auto border-b border-neutral-100">
					{documents.map((doc) => (
						<button
							key={doc.id}
							type="button"
							onClick={() => handleDocSwitch(doc.id)}
							className={`flex-shrink-0 border-b-2 px-3 py-2 text-xs transition-colors ${
								doc.id === activeDocId
									? "border-neutral-800 bg-white text-neutral-800 font-medium"
									: "border-transparent text-neutral-400 hover:text-neutral-600 hover:bg-neutral-50"
							}`}
							title={doc.filename}
						>
							<span className="block max-w-[120px] truncate">
								{doc.filename.replace(".pdf", "")}
							</span>
						</button>
					))}
				</div>
			)}

			{/* Header */}
			<div className="flex items-center justify-between border-b border-neutral-100 px-4 py-3">
				<div className="min-w-0">
					<p className="truncate text-sm font-medium text-neutral-800">
						{document?.filename}
					</p>
					<p className="text-xs text-neutral-400">
						{document?.page_count} page{document?.page_count !== 1 ? "s" : ""}
						{documents.length > 1 && (
							<span>
								{" "}
								· {documents.length} documents in bundle
							</span>
						)}
					</p>
				</div>
			</div>

			{/* Highlight indicator */}
			{highlightText && (
				<div className="flex items-center gap-2 border-b border-yellow-200 bg-yellow-50 px-4 py-1.5">
					<span className="flex-1 truncate text-[11px] text-yellow-800">
						Highlighting: "{highlightText.slice(0, 60)}
						{highlightText.length > 60 ? "…" : ""}"
					</span>
					<button
						type="button"
						onClick={onClearHighlight}
						className="flex-shrink-0 rounded p-0.5 text-yellow-600 hover:bg-yellow-100"
					>
						<X className="h-3.5 w-3.5" />
					</button>
				</div>
			)}

			{/* PDF content */}
			<div className="flex-1 overflow-y-auto p-4">
				{pdfError && (
					<div className="rounded-lg bg-red-50 p-3 text-sm text-red-600">
						{pdfError}
					</div>
				)}

				{pdfUrl && (
					<PDFDocument
						key={document?.id}
						file={pdfUrl}
						onLoadSuccess={({ numPages: pages }) => {
							setNumPages(pages);
							setPdfLoading(false);
							setPdfError(null);
						}}
						onLoadError={(error) => {
							setPdfError(`Failed to load PDF: ${error.message}`);
							setPdfLoading(false);
						}}
						loading={
							<div className="flex items-center justify-center py-12">
								<Loader2 className="h-6 w-6 animate-spin text-neutral-400" />
							</div>
						}
					>
					{!pdfLoading && !pdfError && (
						<Page
							key={`${document?.id}-${currentPage}`}
							pageNumber={currentPage}
							width={pdfPageWidth}
							onRenderTextLayerSuccess={applyHighlightToTextLayer}
							loading={
								<div className="flex items-center justify-center py-12">
									<Loader2 className="h-5 w-5 animate-spin text-neutral-300" />
								</div>
							}
						/>
					)}
					</PDFDocument>
				)}
			</div>

			{/* Page navigation */}
			{numPages > 0 && (
				<div className="flex items-center justify-center gap-3 border-t border-neutral-100 px-4 py-2.5">
					<Button
						variant="ghost"
						size="icon"
						className="h-7 w-7"
						disabled={currentPage <= 1}
						onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
					>
						<ChevronLeft className="h-4 w-4" />
					</Button>
					<span className="text-xs text-neutral-500">
						Page {currentPage} of {numPages}
					</span>
					<Button
						variant="ghost"
						size="icon"
						className="h-7 w-7"
						disabled={currentPage >= numPages}
						onClick={() => setCurrentPage((p) => Math.min(numPages, p + 1))}
					>
						<ChevronRight className="h-4 w-4" />
					</Button>
				</div>
			)}
		</div>
	);
}
