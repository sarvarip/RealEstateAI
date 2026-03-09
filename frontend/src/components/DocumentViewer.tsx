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

function getResponsiveDefaults() {
	const vw = typeof window !== "undefined" ? window.innerWidth : 1440;
	return {
		min: Math.round(Math.max(220, vw * 0.18)),
		max: Math.round(Math.min(700, vw * 0.45)),
		initial: Math.round(Math.max(280, Math.min(vw * 0.3, 500))),
	};
}

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
	targetPage: number | null;
	highlightText: string | null;
	onClearHighlight?: () => void;
}

export function DocumentViewer({
	document,
	targetPage,
	highlightText,
	onClearHighlight,
}: DocumentViewerProps) {
	const [responsiveDefaults] = useState(getResponsiveDefaults);
	const [numPages, setNumPages] = useState<number>(0);
	const [currentPage, setCurrentPage] = useState(1);
	const [pdfLoading, setPdfLoading] = useState(true);
	const [pdfError, setPdfError] = useState<string | null>(null);
	const [width, setWidth] = useState(responsiveDefaults.initial);
	const [dragging, setDragging] = useState(false);
	const containerRef = useRef<HTMLDivElement>(null);
	const scrollTimerRef = useRef<ReturnType<typeof setTimeout>>();
	// Tracks which page the text layer has been rendered for. Prevents
	// applyHighlight from running against a stale text layer during page
	// transitions, which caused the "jump to wrong section" bug.
	const textLayerPageRef = useRef<number | null>(null);
	// Monotonically increasing counter — each applyHighlight call captures
	// the current value and checks it before scrolling. Stale calls (whose
	// captured version != current) skip the scroll, preventing races between
	// competing scrollIntoView animations.
	const highlightVersionRef = useRef(0);

	useEffect(() => {
		if (targetPage !== null && targetPage >= 1 && targetPage <= numPages) {
			setCurrentPage(targetPage);
		}
	}, [targetPage, numPages]);

	const applyHighlightCore = useCallback(() => {
		if (!containerRef.current) return;

		const textLayer = containerRef.current.querySelector(".textLayer");
		if (!textLayer) return;

		textLayer.querySelectorAll(".citation-highlight").forEach((el) => {
			(el as HTMLElement).style.backgroundColor = "";
			(el as HTMLElement).style.borderRadius = "";
			el.classList.remove("citation-highlight");
		});

		if (!highlightText) return;

		const rawSpans = Array.from(
			textLayer.querySelectorAll('span[role="presentation"]'),
		) as HTMLElement[];
		if (rawSpans.length === 0) return;

		const positioned = rawSpans.map((span) => {
			const rect = span.getBoundingClientRect();
			return { span, top: rect.top, left: rect.left, height: rect.height };
		});
		const avgHeight =
			positioned.reduce((s, p) => s + p.height, 0) / positioned.length || 10;
		const lineTolerance = avgHeight * 0.4;
		positioned.sort((a, b) => {
			if (Math.abs(a.top - b.top) < lineTolerance) return a.left - b.left;
			return a.top - b.top;
		});

		const spans = positioned.map((p) => p.span);
		const norms = spans.map((s) =>
			normalizeForMatch(s.textContent || ""),
		);

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
		let firstHighlighted: HTMLElement | null = null;
		for (let i = 0; i < spanRanges.length; i++) {
			if (spanRanges[i].end > matchIdx && spanRanges[i].start < matchEnd) {
				spans[i].style.backgroundColor = "rgba(254, 240, 138, 0.6)";
				spans[i].style.borderRadius = "2px";
				spans[i].classList.add("citation-highlight");
				if (!firstHighlighted) firstHighlighted = spans[i];
			}
		}

		if (firstHighlighted) {
			const target = firstHighlighted;
			const version = ++highlightVersionRef.current;
			clearTimeout(scrollTimerRef.current);
			scrollTimerRef.current = setTimeout(() => {
				if (highlightVersionRef.current !== version) return;
				target.scrollIntoView({ block: "center", behavior: "instant" });
			}, 250);
		}
	}, [highlightText]);

	const applyHighlightRef = useRef(applyHighlightCore);
	applyHighlightRef.current = applyHighlightCore;

	// Called by react-pdf when the text layer finishes rendering.
	// Stable callback (empty deps) so react-pdf doesn't re-render.
	const onTextLayerReady = useCallback(() => {
		textLayerPageRef.current = currentPage;
		applyHighlightRef.current();
	}, [currentPage]);

	// When highlightText changes on the SAME page, onTextLayerReady won't
	// fire (no re-render), so this effect handles that case. The guard
	// ensures we never run against a stale text layer during page transitions.
	useEffect(() => {
		if (textLayerPageRef.current === currentPage) {
			applyHighlightCore();
		}
		return () => clearTimeout(scrollTimerRef.current);
	}, [applyHighlightCore, currentPage]);

	const handleMouseDown = useCallback(
		(e: React.MouseEvent) => {
			e.preventDefault();
			setDragging(true);

			const startX = e.clientX;
			const startWidth = width;

			const handleMouseMove = (moveEvent: MouseEvent) => {
				const delta = startX - moveEvent.clientX;
				const newWidth = Math.min(
					responsiveDefaults.max,
					Math.max(responsiveDefaults.min, startWidth + delta),
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

	// Reset page state when the active document changes (driven by sidebar)
	const prevDocId = useRef(document?.id);
	useEffect(() => {
		if (document?.id !== prevDocId.current) {
			prevDocId.current = document?.id;
			setCurrentPage(1);
			setPdfLoading(true);
			setPdfError(null);
			textLayerPageRef.current = null;
		}
	}, [document?.id]);

	const pdfPageWidth = width - 48;

	if (!document) {
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

			{/* Header */}
			<div className="flex items-center justify-between border-b border-neutral-100 px-4 py-3">
				<div className="min-w-0">
					<p className="truncate text-sm font-medium text-neutral-800">
						{document?.filename}
					</p>
					<p className="text-xs text-neutral-400">
						{document?.page_count} page{document?.page_count !== 1 ? "s" : ""}
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
							onRenderTextLayerSuccess={onTextLayerReady}
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
