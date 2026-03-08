import { FileText, Play } from "lucide-react";
import { useState } from "react";
import type { SectionsProposal } from "../types";

interface SectionProposalProps {
	proposal: SectionsProposal;
	onExecute: (sectionIds: string[]) => void;
	disabled?: boolean;
}

export function SectionProposal({
	proposal,
	onExecute,
	disabled,
}: SectionProposalProps) {
	const [selected, setSelected] = useState<Set<string>>(
		() => new Set(proposal.sections.map((s) => s.id)),
	);

	const toggle = (id: string) => {
		setSelected((prev) => {
			const next = new Set(prev);
			if (next.has(id)) next.delete(id);
			else next.add(id);
			return next;
		});
	};

	const toggleAll = () => {
		if (selected.size === proposal.sections.length) {
			setSelected(new Set());
		} else {
			setSelected(new Set(proposal.sections.map((s) => s.id)));
		}
	};

	return (
		<div className="mx-auto mt-3 max-w-2xl rounded-xl border border-blue-200 bg-blue-50/50 p-4">
			<div className="mb-3 flex items-center gap-2">
				<FileText className="h-4 w-4 text-blue-600" />
				<span className="text-sm font-medium text-blue-900">
					Report Sections
				</span>
				<span className="text-xs text-blue-600">
					Select which sections to generate
				</span>
			</div>

			<div className="mb-3 space-y-1.5">
				{proposal.sections.map((sec) => (
					<label
						key={sec.id}
						className="flex cursor-pointer items-center gap-2.5 rounded-lg border border-transparent px-2 py-1.5 transition-colors hover:border-blue-200 hover:bg-blue-100/50"
					>
						<input
							type="checkbox"
							checked={selected.has(sec.id)}
							onChange={() => toggle(sec.id)}
							className="h-3.5 w-3.5 rounded border-blue-300 text-blue-600 focus:ring-blue-500"
						/>
						<span className="text-sm font-medium text-neutral-800">
							{sec.title}
						</span>
						<span className="text-xs text-neutral-500">
							— {sec.description}
						</span>
					</label>
				))}
			</div>

			<div className="flex items-center justify-between border-t border-blue-200 pt-3">
				<button
					type="button"
					onClick={toggleAll}
					className="text-xs text-blue-600 hover:text-blue-800"
				>
					{selected.size === proposal.sections.length
						? "Deselect all"
						: "Select all"}
				</button>

				<button
					type="button"
					disabled={selected.size === 0 || disabled}
					onClick={() => onExecute(Array.from(selected))}
					className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
				>
					<Play className="h-3.5 w-3.5" />
					Generate {selected.size} section{selected.size !== 1 ? "s" : ""}
				</button>
			</div>
		</div>
	);
}
