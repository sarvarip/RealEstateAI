import { motion } from "framer-motion";
import {
	AlertTriangle,
	Bot,
	CheckCircle2,
	ExternalLink,
} from "lucide-react";
import type { ReactNode } from "react";
import { Streamdown } from "streamdown";
import "streamdown/styles.css";
import type { AnswerSegment, Citation, Message } from "../types";

interface MessageBubbleProps {
	message: Message;
	onCitationClick?: (citation: Citation) => void;
	onRetry?: () => void;
}

function CitationChip({
	citation,
	onClick,
}: { citation: Citation; onClick?: (c: Citation) => void }) {
	const label = `${citation.filename.replace(".pdf", "")}, p.${citation.page}`;
	return (
		<button
			type="button"
			onClick={() => onClick?.(citation)}
			title={
				citation.verified
					? `Verified: "${citation.quote}"`
					: "Could not verify this citation against the source document"
			}
			className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium leading-tight transition-colors align-middle ${
				citation.verified
					? "border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100"
					: "border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100"
			}`}
		>
			{citation.verified ? (
				<CheckCircle2 className="h-3 w-3" />
			) : (
				<AlertTriangle className="h-3 w-3" />
			)}
			<span className="max-w-[140px] truncate">{label}</span>
			<ExternalLink className="h-2.5 w-2.5 opacity-50" />
		</button>
	);
}

function SegmentedContent({
	segments,
	onCitationClick,
}: {
	segments: AnswerSegment[];
	onCitationClick?: (c: Citation) => void;
}) {
	const nodes: ReactNode[] = [];

	for (let i = 0; i < segments.length; i++) {
		const seg = segments[i];
		const trimmed = seg.text.trim();

		if (trimmed) {
			nodes.push(
				<div key={`seg-${i}`} className="prose inline">
					<Streamdown>{trimmed}</Streamdown>
				</div>,
			);
		}

		for (let j = 0; j < seg.citations.length; j++) {
			nodes.push(
				<CitationChip
					key={`cite-${i}-${j}`}
					citation={seg.citations[j]}
					onClick={onCitationClick}
				/>,
			);
		}

		if (seg.citations.length > 0 && i < segments.length - 1) {
			nodes.push(<span key={`sp-${i}`}>{" "}</span>);
		}
	}

	return <>{nodes}</>;
}

function FlatContent({
	content,
	citations,
	onCitationClick,
}: {
	content: string;
	citations: Citation[];
	onCitationClick?: (c: Citation) => void;
}) {
	return (
		<>
			<div className="prose">
				<Streamdown>{content}</Streamdown>
			</div>
			{citations.length > 0 && (
				<div className="mt-2 flex flex-wrap gap-1">
					{citations.map((cite) => (
						<CitationChip
							key={cite.index}
							citation={cite}
							onClick={onCitationClick}
						/>
					))}
				</div>
			)}
		</>
	);
}

export function MessageBubble({
	message,
	onCitationClick,
	onRetry,
}: MessageBubbleProps) {
	const citations = message.citations ?? [];
	const segments = message.segments;
	const hasUnverified = citations.some((c) => !c.verified);

	if (message.role === "system") {
		return (
			<motion.div
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ duration: 0.2 }}
				className="flex justify-center py-2"
			>
				<p className="text-xs text-neutral-400">{message.content}</p>
			</motion.div>
		);
	}

	if (message.role === "user") {
		return (
			<motion.div
				initial={{ opacity: 0, y: 8 }}
				animate={{ opacity: 1, y: 0 }}
				transition={{ duration: 0.2 }}
				className="flex justify-end py-1.5"
			>
				<div className="max-w-[75%] rounded-2xl rounded-br-md bg-neutral-100 px-4 py-2.5">
					<p className="whitespace-pre-wrap text-sm text-neutral-800">
						{message.content}
					</p>
				</div>
			</motion.div>
		);
	}

	return (
		<motion.div
			initial={{ opacity: 0, y: 8 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ duration: 0.2 }}
			className="flex gap-3 py-1.5"
		>
			<div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-neutral-900">
				<Bot className="h-4 w-4 text-white" />
			</div>
			<div className="min-w-0 max-w-[80%]">
				{segments && segments.length > 0 ? (
					<SegmentedContent
						segments={segments}
						onCitationClick={onCitationClick}
					/>
				) : citations.length > 0 ? (
					<FlatContent
						content={message.content}
						citations={citations}
						onCitationClick={onCitationClick}
					/>
				) : (
					<div className="prose">
						<Streamdown>{message.content}</Streamdown>
					</div>
				)}

				{hasUnverified && (
					<div className="mt-2 flex items-center gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-1.5">
						<AlertTriangle className="h-3.5 w-3.5 flex-shrink-0 text-amber-500" />
						<span className="text-[11px] text-amber-700">
							Some citations could not be verified against the source documents
						</span>
						{onRetry && (
							<button
								type="button"
								onClick={onRetry}
								className="ml-auto text-[11px] font-medium text-amber-700 underline hover:text-amber-900"
							>
								Retry
							</button>
						)}
					</div>
				)}

				{citations.length > 0 && !hasUnverified && (
					<p className="mt-1.5 text-xs text-neutral-400">
						{citations.length} source
						{citations.length !== 1 ? "s" : ""} cited
					</p>
				)}
			</div>
		</motion.div>
	);
}

export function ThinkingBubble({ toolStatus }: { toolStatus?: string | null }) {
	return (
		<div className="flex gap-3 py-1.5">
			<div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-neutral-900">
				<Bot className="h-4 w-4 text-white" />
			</div>
			<div className="min-w-0 max-w-[80%]">
				{toolStatus ? (
					<div className="flex items-center gap-2 py-2">
						<span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-400" />
						<span className="text-xs text-neutral-500 italic">{toolStatus}</span>
					</div>
				) : (
					<div className="flex items-center gap-1 py-2">
						<span className="h-1.5 w-1.5 animate-pulse rounded-full bg-neutral-400" />
						<span
							className="h-1.5 w-1.5 animate-pulse rounded-full bg-neutral-400"
							style={{ animationDelay: "0.15s" }}
						/>
						<span
							className="h-1.5 w-1.5 animate-pulse rounded-full bg-neutral-400"
							style={{ animationDelay: "0.3s" }}
						/>
					</div>
				)}
			</div>
		</div>
	);
}
