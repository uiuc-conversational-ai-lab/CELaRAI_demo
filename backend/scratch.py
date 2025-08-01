question_types = ["Math", "Science", "History"]
difficulty = 5
c

additional_instructions=f"Question types: {', '.join(question_types)}.\nDifficulty (out of 10; 1 = Elementary; 5 = Undergraduate; 10 = Expert): {difficulty}.\nCustom instructions: {custom_instructions or "None"}"
print()