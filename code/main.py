"""
Multi-Domain Support Triage System

Architecture with 3-step pipeline:
1. Vector retrieval (FAISS + Sentence-BERT)
2. Unified agent (Groq LLM with RAG context)
3. Output validation
"""

import os
import pandas as pd
import time
import gc
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Get absolute paths relative to this script
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

from vector_store import VectorStore
from unified_agent import UnifiedAgent


class SupportTriageSystem:
    """Main orchestration system for support ticket triage."""

    def __init__(self, data_dir: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        """Initialize the triage system with all components."""
        print("Initializing Support Triage System...")

        # Load environment variables from code directory
        load_dotenv(SCRIPT_DIR / '.env')

        # Verify API key
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in environment")

        # Set default data directory
        if data_dir is None:
            data_dir = str(PROJECT_ROOT / "data")

        # Initialize components
        print("\n1. Initializing vector store...")
        self.vector_store = VectorStore(data_dir=data_dir)
        self.vector_store.index_documents()

        # Force garbage collection after indexing
        gc.collect()

        print("\n2. Initializing unified agent...")
        self.agent = UnifiedAgent(model=model)

        print("\n✓ System ready\n")

    def process_ticket(
        self,
        issue: str,
        subject: str,
        company: Optional[str]
    ) -> Dict:
        """
        Process a single support ticket through the 3-step pipeline.

        Args:
            issue: The ticket content
            subject: The ticket subject
            company: Company hint (may be None or "None")

        Returns:
            Dictionary with all outputs
        """
        # Normalize company
        company_name = None if company == "None" or not company else company

        # Step 1: Vector retrieval
        query = f"{subject} {issue}"
        retrieved_docs = self.vector_store.retrieve(
            query=query,
            company=company_name,
            top_k=3,
            min_similarity=0.2
        )

        # If no relevant docs found, still try with agent
        # Don't auto-escalate - let the agent decide

        # Step 2: Unified agent (single LLM call with retry)
        ticket_data = {
            'Issue': issue,
            'Subject': subject,
            'Company': company_name or 'Unknown'
        }

        result = self._call_agent_with_retry(ticket_data, retrieved_docs)

        # Step 3: Output validation (already done in unified_agent)
        return {
            'status': result['status'],
            'product_area': result['product_area'],
            'response': result['response'],
            'request_type': result['request_type']
        }

    def _call_agent_with_retry(
        self,
        ticket: Dict,
        retrieved_docs: list,
        max_retries: int = 3
    ) -> Dict:
        """Call unified agent with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return self.agent.process_ticket(ticket, retrieved_docs)
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"\nRate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt == max_retries - 1:
                    # Final attempt failed, escalate
                    print(f"\nAgent call failed after {max_retries} attempts: {e}")
                    break
                else:
                    # Non-rate-limit error, escalate immediately
                    print(f"\nAgent call error: {e}")
                    break

        # All retries failed, return escalation
        return {
            'status': 'Escalated',
            'product_area': 'general',
            'response': '',
            'request_type': 'product_issue'
        }

    def process_csv(
        self,
        input_path: str,
        output_path: str,
        limit: Optional[int] = None,
        batch_size: int = 5
    ):
        """
        Process a CSV file of support tickets with memory-efficient batching.

        Args:
            input_path: Path to input CSV
            output_path: Path to output CSV
            limit: Optional limit on number of tickets to process
            batch_size: Process tickets in batches and cleanup memory (default: 5)
        """
        print(f"Processing tickets from: {input_path}")

        # Load input CSV
        df = pd.read_csv(input_path)

        if limit:
            df = df.head(limit)
            print(f"Processing first {limit} tickets")

        print(f"Total tickets to process: {len(df)}")
        print(f"Batch size: {batch_size} (memory optimization)")

        # Process each ticket
        results = []
        output_df = pd.DataFrame()  # Initialize in case df is empty

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing tickets"):
            ticket_num = int(idx) if isinstance(idx, (int, float)) else len(results)

            try:
                result = self.process_ticket(
                    issue=str(row['Issue']),
                    subject=str(row.get('Subject', '')),
                    company=str(row.get('Company', 'None')) if row.get('Company') else None
                )

                results.append({
                    'Issue': row['Issue'],
                    'Subject': row.get('Subject', ''),
                    'Company': row.get('Company', 'None'),
                    'Response': result['response'],
                    'Product Area': result['product_area'],
                    'Status': result['status'],
                    'Request Type': result['request_type']
                })

            except Exception as e:
                print(f"\nError processing ticket {ticket_num}: {e}")
                results.append({
                    "Issue": row['Issue'],
                    "Subject": row.get('Subject', ''),
                    "Company": row.get('Company', 'None'),
                    "Response": "Escalate to a human",
                    "Product Area": "unknown",
                    "Status": "Escalated",
                    "Request Type": "invalid"
                })

            # Save results incrementally
            output_df = pd.DataFrame(results)
            output_df.to_csv(output_path, index=False, quoting=1)

            # Memory cleanup every batch_size tickets
            if (ticket_num + 1) % batch_size == 0:
                gc.collect()

        # Final cleanup
        gc.collect()

        print("\n\n✓ Results saved to:", output_path)

        # Print summary statistics
        self._print_summary(output_df)

    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics of processed tickets."""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)

        print(f"\nTotal tickets processed: {len(df)}")

        print("\nStatus breakdown:")
        status_counts = df['Status'].value_counts()
        for status, count in status_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {status}: {count} ({pct:.1f}%)")

        print("\nRequest type breakdown:")
        type_counts = df['Request Type'].value_counts()
        for req_type, count in type_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {req_type}: {count} ({pct:.1f}%)")

        if 'Company' in df.columns:
            print("\nCompany breakdown:")
            company_counts = df['Company'].value_counts()
            for company, count in company_counts.items():
                pct = (count / len(df)) * 100
                print(f"  {company}: {count} ({pct:.1f}%)")

        print("\n" + "="*60)


def main():
    """Main entry point for the support triage system."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Domain Support Triage System"
    )
    parser.add_argument(
        '--input',
        type=str,
        default=str(PROJECT_ROOT / 'support_tickets' / 'support_tickets.csv'),
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(PROJECT_ROOT / 'support_tickets' / 'output.csv'),
        help='Output CSV file path'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of tickets to process (for testing)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to support documentation directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llama-3.1-8b-instant',
        help='Groq model to use (llama-3.3-70b-versatile or llama-3.1-8b-instant)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Batch size for memory cleanup (lower = less memory usage)'
    )

    args = parser.parse_args()

    # Initialize system
    system = SupportTriageSystem(data_dir=args.data_dir, model=args.model)

    # Process tickets
    system.process_csv(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
