from datetime import datetime
import pandas as pd
import earningscall
from earningscall import get_company
from pathlib import Path
import json
import os

class Company:
    def __init__(self, company_code: str):
        self.company_code = company_code.lower()
        self.company = self._initialize_company()

    def _initialize_company(self) -> earningscall.company.Company:
        """
        Initializes the company object using the given company code.
        """
        return get_company(self.company_code)

    def get_and_save_one_transcript(self, base_path: str, year: int, quarter: int, level=3) -> None:
        """
        Fetches and saves a single earnings call transcript for the specified year and quarter.
        """
        print(f"Fetching transcript for {self.company_code.upper()} Q{quarter} {year}")
        transcript = self.company.get_transcript(year=year, quarter=quarter, level=level)

        path = os.path.join(base_path, self.company_code.upper(), str(year), f"Q{quarter}")

        if transcript:
            self.save_transcript(transcript, path)
            print(f"Transcript found and loaded. Q{quarter} {year}. [OK]")
        else:
            print(f"No transcript found. Q{quarter} {year}. [ERROR]")

    def get_and_save_all_transcripts_and_audio(self, base_path: str, level=4) -> None:
        """
        Fetches and saves all available earnings call transcripts for the company.
        """
        print(f"Fetching all transcripts for {self.company_code.upper()}..")

        for event in self.company.events():
            # Skip future events
            if datetime.now().timestamp() < event.conference_date.timestamp():
                print(f"* {self.company.company_info.symbol} Q{event.quarter} {event.year} -- skipping, conference date in the future")
                continue
            transcripts = {}

            try:
                try:
                    transcript_level_4 = self.company.get_transcript(event=event, level=4)
                    transcripts['LEVEL_4'] = transcript_level_4
                except Exception as e:
                    print(f"Failed to retrieve LEVEL_4 transcript: {e}")
                try:
                    transcript_level_3 = self.company.get_transcript(event=event, level=3)
                    transcripts['LEVEL_3'] = transcript_level_3
                except Exception as e:
                    print(f"Failed to retrieve LEVEL_3 transcript: {e}")

                path = os.path.join(base_path, self.company_code.upper(), str(event.year), f"Q{event.quarter}")

                if transcripts:
                    self.save_transcripts(transcripts, path)
                    self.company.download_audio_file(event=event, file_name=path + '/audio.mp3')
                    print(f"Transcript and audio found and loaded. Q{event.quarter} {event.year}. [OK]")
                else:
                    print(f"No transcript found. Q{event.quarter} {event.year}. [NOT FOUND]")
            except Exception as e:
                print(f"Error processing Q{event.quarter} {event.year} for {self.company_code.upper()}: {e}")
        print("-" * 100)


    def save_transcripts(self, transcripts:dict, path: str):
        self.save_transcripts_json(transcripts, path)
        self.save_transcript_csv(transcripts, path)
        
    
    def save_transcripts_json(self, transcripts: dict, path: str) -> None:
        """
        Saves the transcript dictionary to a JSON file.
        """
        # Ensure the directory exists
        Path(path).mkdir(parents=True, exist_ok=True)
        
        for level, transcript in transcripts.items():
            file_path = os.path.join(path, f'{level}.json')
            try:
                with open(file_path, "w", encoding="utf-8") as archivo:
                    json.dump(transcript.to_dict(), archivo, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Failed to save transcript of level ({level}): {e}")

    def save_transcript_csv(self, transcripts: dict, path: str) -> None:
        """
        Converts transcript data to CSV and saves it.
        """
        csv_path = os.path.join(path, "transcript.csv")

        speakers = transcripts['LEVEL_3'].to_dict().get("speakers", [])
        rows = []
        for speaker in speakers:

            speaker_info = speaker.get("speaker_info", {})
            name = speaker_info.get("name", None) if speaker_info else None
            title = speaker_info.get("title", None) if speaker_info else None

            row = {
                    "speaker_id": speaker["speaker"],
                    "name": name,
                    "title": title,
                    "text": speaker["text"],
                    "start_time": speaker.get("start_times", [None])[0],
                    "end_time": speaker.get("start_times", [None])[-1]}
            rows.append(row)

        df = pd.DataFrame(rows)
        try:
            df.to_csv(csv_path, index=False, encoding="utf-8")
        except Exception as e:
            print(f"Failed to save transcript CSV: {e}")

