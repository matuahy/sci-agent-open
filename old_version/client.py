# ./client.py 客户端
"""
客户端
"""
import requests
import json

API_URL = "http://localhost:8000/query"

def main():
    original_query = None
    while True:
        if original_query is None:
            query = input("Enter your scientific query (or 'quit' to exit): ").strip()
            if query.lower() == 'quit':
                print("Exiting...")
                break
            if not query:
                print("Query cannot be empty. Try again.")
                continue
            original_query = query
            current_query = query
        else:
            current_query = original_query

        data = {"query": current_query}

        try:
            response = requests.post(API_URL, json=data)
            response.raise_for_status()
            result = response.json()

            planner_output = result.get("planner_output", {})
            print("\n=== Response ===")
            print("Planner Output:")
            print(json.dumps(planner_output, indent=2, ensure_ascii=False))

            print("\nSearch Results:")
            print(json.dumps(result.get("search_results", []), indent=2, ensure_ascii=False))
            print("\n")

            #----------------------------
            # 在打印完 search_results 之后
            print("\nRAG DB (chroma_dir):")
            print(json.dumps(result.get("chroma_dir", []), indent=2, ensure_ascii=False))
            #------------------------------

            clarifying_questions = planner_output.get("clarifying_questions", [])
            if clarifying_questions:
                print("\nCLARIFYING QUESTIONS:")
                print("=" * 60)
                answers = []
                for i, question in enumerate(clarifying_questions, 1):
                    print(f"\nQ{i}: {question}")
                    answer = input(f"A{i}: ").strip()
                    answers.append(answer if answer else "No preference")
                    print(f" {answers[-1]}")
                print("\nANSWERS:")
                print("=" * 60)
                for i, (q, a) in enumerate(zip(clarifying_questions, answers), 1):
                    print(f"Q{i}: {a}")

                answers_str = "\n".join(
                    [f"Q{i}: {q}\nA{i}: {a}" for i, (q, a) in enumerate(zip(clarifying_questions, answers), 1)])
                new_query = f"{original_query}\n\nAnswers to clarifying questions:\n{answers_str}"
                print(f"\nUpdated query sent")
                original_query = new_query
                print("\nGenerating full plan...\n")
                continue
            else:
                print("Research plan completed!")

                # ==================================================================
                # ONLY ADDED PART STARTS HERE - Beautiful Final Report Display
                # ==================================================================
                final_report = result.get("final_report")
                if final_report:
                    print("\n" + "="*80)
                    print(" FULL LITERATURE REVIEW GENERATED ".center(80, "="))
                    print("="*80)
                    print(f"Title: {final_report.get('title', 'Untitled Review')}")
                    print(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Sections: {len(final_report.get('sections', []))}\n")

                    def show_section(sec, depth=0):
                        indent = "  " * depth
                        title = sec.get("section_title", "Untitled")
                        content = sec.get("content", "").strip()
                        print(f"{indent}• {title}")
                        if content:
                            lines = content.split("\n")[:8]  # Show first 8 lines
                            for line in lines:
                                if line.strip():
                                    print(f"{indent}  {line.strip()}")
                            if len(content.split("\n")) > 8:
                                print(f"{indent}  ... (full content {len(content.splitlines())} lines)")
                            print()
                        for sub in sec.get("subsections", []):
                            show_section(sub, depth + 1)

                    for section in final_report.get("sections", []):
                        show_section(section)

                    print("Full review completed successfully!")
                    print("You can now copy the above content or wait for PDF export feature.")
                    print("="*80)
                else:
                    print("\nWarning: final_report not received (workflow may be incomplete)")
                # ==================================================================
                # ONLY ADDED PART ENDS HERE
                # ==================================================================

                original_query = None  # Reset for next new query

        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            original_query = None

if __name__ == "__main__":
    main()