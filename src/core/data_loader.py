import os
import time
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import pandas as pd
import pickle
import sys

load_dotenv()

class GoogleBooksLoader():
    """Loader that fetches books from Google Books API systematically"""
    
    def __init__(self):
        self.BASE_URL = "https://www.googleapis.com/books/v1/volumes"
        self.RESULTS_PER_PAGE = 40
        self.MIN_BOOKS_PER_VALUE = 100
        self.MIN_DESCRIPTION_LENGTH = 2048
        self.API_KEY = os.getenv("GOOGLE_API_KEY")
        
        # Paths for incremental saving
        self.PICKLE_PATH = "../data/books_data.pkl"
        self.COUNTS_PICKLE_PATH = "../data/value_counts.pkl"
        
        self.SEARCH_STRATEGIES = [
            {
                "param": "subject", 
                "values": [
                    # Fiction & Literature
                    "Fiction", "Literature", "Romance", "Mystery", "Thriller", "Fantasy", "Science Fiction", 
                    "Horror", "Adventure", "Young Adult", "Children's Literature", "Poetry", "Drama",
                    
                    # Non-Fiction Categories
                    "Science", "History", "Biography", "Technology", "Art", "Philosophy", "Psychology", 
                    "Business", "Travel", "Cooking", "Health", "Self-Help", "Religion", "Politics",
                    "Economics", "Education", "Sports", "Music", "Film", "Photography", "Architecture",
                    "Mathematics", "Physics", "Chemistry", "Biology", "Medicine", "Engineering",
                    "Computer Science", "Law", "Military", "Geography", "Archaeology", "Anthropology",
                    "Sociology", "Linguistics", "Astronomy", "Geology", "Botany", "Zoology", "Genetics",
                    "Neuroscience", "Economics", "Finance", "Marketing", "Management", "Leadership",
                    "Entrepreneurship", "Investing", "Real Estate", "Cooking", "Gardening", "Fitness",
                    "Yoga", "Meditation", "Parenting", "Relationships", "Career", "Personal Development",
                    "Motivation", "Productivity", "Creativity", "Design", "Fashion", "Beauty", "Home",
                    "DIY", "Crafts", "Hobbies", "Games", "Puzzles", "Collecting", "Outdoors", "Nature",
                    "Wildlife", "Environment", "Climate", "Sustainability", "Recycling", "Organic",
                    "Vegan", "Vegetarian", "Gluten-Free", "Keto", "Paleo", "Mediterranean", "Asian",
                    "Italian", "French", "Mexican", "Indian", "Middle Eastern", "African", "Caribbean",
                    "Latin American", "European", "American", "British", "Canadian", "Australian"
                ]
            },
            {
                "param": "inauthor", 
                "values": [
                    # Popular Fiction Authors
                    "Stephen King", "J.K. Rowling", "J.R.R. Tolkien", "Agatha Christie", "Toni Morrison",
                    "Ernest Hemingway", "F. Scott Fitzgerald", "Jane Austen", "Charles Dickens",
                    "Mark Twain", "Harper Lee", "George Orwell", "Aldous Huxley", "Ray Bradbury",
                    "Isaac Asimov", "Arthur C. Clarke", "Philip K. Dick", "Ursula K. Le Guin",
                    "Margaret Atwood", "Neil Gaiman", "Terry Pratchett", "Brandon Sanderson",
                    "Patrick Rothfuss", "George R.R. Martin", "J.R.R. Tolkien", "C.S. Lewis",
                    "J.D. Salinger", "Kurt Vonnegut", "John Steinbeck", "William Faulkner",
                    "Virginia Woolf", "James Joyce", "Gabriel Garcia Marquez", "Milan Kundera",
                    "Umberto Eco", "Italo Calvino", "Jorge Luis Borges", "Pablo Neruda",
                    "Octavio Paz", "Carlos Fuentes", "Isabel Allende", "Mario Vargas Llosa",
                    
                    # Contemporary Authors
                    "Yuval Noah Harari", "Michelle Obama", "Malcolm Gladwell", "Bill Bryson",
                    "Mary Roach", "Oliver Sacks", "Atul Gawande", "Siddhartha Mukherjee",
                    "Steven Pinker", "Daniel Kahneman", "Nassim Nicholas Taleb", "Jared Diamond",
                    "Yuval Noah Harari", "Sapiens", "Homo Deus", "21 Lessons for the 21st Century",
                    "Angela Duckworth", "Carol Dweck", "Brené Brown", "Simon Sinek", "Seth Godin",
                    "Tim Ferriss", "Ryan Holiday", "Robert Greene", "Cal Newport", "James Clear",
                    "Atomic Habits", "Deep Work", "Digital Minimalism", "So Good They Can't Ignore You",
                    
                    # Business & Leadership Authors
                    "Dale Carnegie", "Stephen Covey", "Jim Collins", "Patrick Lencioni",
                    "Chip Heath", "Dan Heath", "Made to Stick", "Switch", "The Power of Moments",
                    "Daniel H. Pink", "Drive", "To Sell Is Human", "When", "A Whole New Mind",
                    "Adam Grant", "Give and Take", "Originals", "Think Again", "Hidden Potential",
                    "Simon Sinek", "Start With Why", "Leaders Eat Last", "The Infinite Game",
                    "Seth Godin", "Purple Cow", "Tribes", "Linchpin", "The Dip", "This Is Marketing",
                    
                    # Science & Technology Authors
                    "Richard Dawkins", "The Selfish Gene", "The Blind Watchmaker", "The God Delusion",
                    "Carl Sagan", "Cosmos", "Contact", "The Demon-Haunted World", "Pale Blue Dot",
                    "Neil deGrasse Tyson", "Astrophysics for People in a Hurry", "Death by Black Hole",
                    "Brian Greene", "The Elegant Universe", "The Fabric of the Cosmos", "The Hidden Reality",
                    "Michio Kaku", "Physics of the Impossible", "Physics of the Future", "The Future of the Mind",
                    "Sean Carroll", "The Big Picture", "Something Deeply Hidden", "The Particle at the End of the Universe",
                    "Lisa Randall", "Warped Passages", "Knocking on Heaven's Door", "Dark Matter and the Dinosaurs",
                    "Max Tegmark", "Our Mathematical Universe", "Life 3.0", "The Mathematical Universe",
                    
                    # History & Biography Authors
                    "Doris Kearns Goodwin", "Team of Rivals", "The Bully Pulpit", "Leadership in Turbulent Times",
                    "David McCullough", "John Adams", "Truman", "The Wright Brothers", "1776",
                    "Ron Chernow", "Alexander Hamilton", "Washington: A Life", "Grant", "Titan",
                    "Walter Isaacson", "Steve Jobs", "Einstein", "Leonardo da Vinci", "Benjamin Franklin",
                    "Doris Kearns Goodwin", "Team of Rivals", "The Bully Pulpit", "Leadership in Turbulent Times",
                    "Erik Larson", "The Devil in the White City", "In the Garden of Beasts", "Dead Wake",
                    "Bill Bryson", "A Short History of Nearly Everything", "At Home", "The Body",
                    "Mary Roach", "Stiff", "Gulp", "Grunt", "Packing for Mars", "Fuzz",
                    
                    # Psychology & Self-Help Authors
                    "Daniel Kahneman", "Thinking, Fast and Slow", "Noise", "The Undoing Project",
                    "Nassim Nicholas Taleb", "The Black Swan", "Antifragile", "Skin in the Game",
                    "Malcolm Gladwell", "The Tipping Point", "Blink", "Outliers", "David and Goliath",
                    "Angela Duckworth", "Grit", "The Power of Passion and Perseverance",
                    "Carol Dweck", "Mindset", "The New Psychology of Success",
                    "Brené Brown", "The Gifts of Imperfection", "Daring Greatly", "Rising Strong",
                    "Simon Sinek", "Start With Why", "Leaders Eat Last", "The Infinite Game",
                    "Cal Newport", "Deep Work", "Digital Minimalism", "So Good They Can't Ignore You",
                    "James Clear", "Atomic Habits", "Tiny Changes, Remarkable Results"
                ]
            },
            {
                "param": "intitle", 
                "values": [
                    # Technology & Programming
                    "Python", "Machine Learning", "Artificial Intelligence", "Data Science", "Deep Learning",
                    "Neural Networks", "Computer Vision", "Natural Language Processing", "Robotics",
                    "Blockchain", "Cryptocurrency", "Bitcoin", "Ethereum", "Web Development",
                    "JavaScript", "React", "Node.js", "Angular", "Vue.js", "TypeScript",
                    "Java", "C++", "C#", "Go", "Rust", "Swift", "Kotlin", "Scala",
                    "Database", "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "Redis",
                    "Cloud Computing", "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
                    "DevOps", "CI/CD", "Git", "GitHub", "Agile", "Scrum", "Kanban",
                    
                    # Science & Research
                    "Quantum Physics", "Relativity", "String Theory", "Big Bang", "Dark Matter",
                    "Dark Energy", "Black Holes", "Exoplanets", "Mars", "Space Exploration",
                    "Climate Change", "Global Warming", "Renewable Energy", "Solar Power",
                    "Wind Energy", "Nuclear Energy", "Fossil Fuels", "Carbon Footprint",
                    "Biodiversity", "Conservation", "Sustainability", "Green Technology",
                    "Genetics", "DNA", "CRISPR", "Gene Editing", "Evolution", "Natural Selection",
                    "Neuroscience", "Brain", "Consciousness", "Memory", "Learning", "Psychology",
                    "Behavioral Economics", "Decision Making", "Cognitive Bias", "Heuristics",
                    
                    # History & World Events
                    "World War", "World War I", "World War II", "Cold War", "Vietnam War",
                    "Korean War", "Civil War", "American Revolution", "French Revolution",
                    "Industrial Revolution", "Renaissance", "Enlightenment", "Middle Ages",
                    "Ancient History", "Ancient Rome", "Ancient Greece", "Ancient Egypt",
                    "Ancient China", "Ancient India", "Maya", "Aztec", "Inca", "Vikings",
                    "Crusades", "Black Death", "Plague", "Spanish Flu", "Great Depression",
                    "Space Race", "Moon Landing", "Apollo", "NASA", "Cold War", "Berlin Wall",
                    "Fall of Communism", "9/11", "Terrorism", "War on Terror", "Arab Spring",
                    
                    # Health & Wellness
                    "Mental Health", "Depression", "Anxiety", "Stress", "Mindfulness", "Meditation",
                    "Yoga", "Exercise", "Fitness", "Nutrition", "Diet", "Weight Loss",
                    "Healthy Eating", "Superfoods", "Vitamins", "Supplements", "Herbs",
                    "Alternative Medicine", "Traditional Medicine", "Chinese Medicine",
                    "Ayurveda", "Homeopathy", "Naturopathy", "Chiropractic", "Acupuncture",
                    "Sleep", "Insomnia", "Dreams", "Circadian Rhythm", "Hormones",
                    "Gut Health", "Microbiome", "Probiotics", "Prebiotics", "Inflammation",
                    "Autoimmune", "Cancer", "Heart Disease", "Diabetes", "Obesity",
                    
                    # Business & Economics
                    "Entrepreneurship", "Startup", "Business Strategy", "Marketing", "Sales",
                    "Leadership", "Management", "Team Building", "Communication", "Negotiation",
                    "Finance", "Investing", "Stock Market", "Real Estate", "Cryptocurrency",
                    "Bitcoin", "Ethereum", "Blockchain", "NFT", "DeFi", "Fintech",
                    "E-commerce", "Digital Marketing", "Social Media", "Content Marketing",
                    "Branding", "Customer Experience", "User Experience", "Design Thinking",
                    "Innovation", "Disruption", "Digital Transformation", "Remote Work",
                    "Work From Home", "Productivity", "Time Management", "Goal Setting",
                    
                    # Philosophy & Religion
                    "Philosophy", "Ethics", "Morality", "Existentialism", "Stoicism",
                    "Buddhism", "Zen", "Taoism", "Confucianism", "Hinduism", "Islam",
                    "Christianity", "Judaism", "Atheism", "Agnosticism", "Spirituality",
                    "Consciousness", "Free Will", "Determinism", "Pragmatism", "Utilitarianism",
                    "Virtue Ethics", "Deontology", "Epistemology", "Metaphysics", "Logic",
                    "Critical Thinking", "Fallacies", "Argumentation", "Rhetoric",
                    
                    # Arts & Culture
                    "Art History", "Renaissance Art", "Modern Art", "Contemporary Art",
                    "Impressionism", "Cubism", "Surrealism", "Abstract Expressionism",
                    "Photography", "Cinema", "Film History", "Hollywood", "Independent Film",
                    "Documentary", "Music History", "Classical Music", "Jazz", "Rock",
                    "Pop Music", "Hip Hop", "Electronic Music", "World Music", "Folk Music",
                    "Literature", "Poetry", "Drama", "Theater", "Dance", "Ballet",
                    "Architecture", "Design", "Fashion", "Fashion History", "Style",
                    "Beauty", "Cosmetics", "Skincare", "Hair Care", "Makeup"
                ]
            },
            {
                "param": "isbn", 
                "values": [
                    # Popular book series and publishers
                    "978014", "978006", "978031", "978034", "978038", "978039", "978044", "978045",
                    "978052", "978055", "978059", "978061", "978067", "978068", "978069", "978074",
                    "978081", "978082", "978083", "978084", "978085", "978086", "978087", "978088",
                    "978089", "978090", "978091", "978092", "978093", "978094", "978095", "978096",
                    "978097", "978098", "978099", "978100", "978101", "978102", "978103", "978104",
                    "978105", "978106", "978107", "978108", "978109", "978110", "978111", "978112",
                    "978113", "978114", "978115", "978116", "978117", "978118", "978119", "978120",
                    "978121", "978122", "978123", "978124", "978125", "978126", "978127", "978128",
                    "978129", "978130", "978131", "978132", "978133", "978134", "978135", "978136",
                    "978137", "978138", "978139", "978140", "978141", "978142", "978143", "978144",
                    "978145", "978146", "978147", "978148", "978149", "978150", "978151", "978152",
                    "978153", "978154", "978155", "978156", "978157", "978158", "978159", "978160",
                    "978161", "978162", "978163", "978164", "978165", "978166", "978167", "978168",
                    "978169", "978170", "978171", "978172", "978173", "978174", "978175", "978176",
                    "978177", "978178", "978179", "978180", "978181", "978182", "978183", "978184",
                    "978185", "978186", "978187", "978188", "978189", "978190", "978191", "978192",
                    "978193", "978194", "978195", "978196", "978197", "978198", "978199", "978200"
                ]
            }
        ]
        
        # Load existing progress if available
        self.books_data = self._load_pickle(self.PICKLE_PATH) or {}  # book_id -> book_data
        self.value_counts = self._load_pickle(self.COUNTS_PICKLE_PATH) or {}  # (param, value) -> count
        
        # Print progress summary if resuming
        if self.books_data:
            print(f"📚 Resuming from previous session: {len(self.books_data)} books already collected")

    def _save_progress(self):
        """Save current progress to pickle files"""
        try:
            with open(self.PICKLE_PATH, "wb") as f:
                pickle.dump(self.books_data, f)
            with open(self.COUNTS_PICKLE_PATH, "wb") as f:
                pickle.dump(self.value_counts, f)
            print(f"💾 Progress saved: {len(self.books_data)} books")
        except Exception as e:
            print(f"⚠️ Warning: Could not save progress: {e}")

    def _load_pickle(self, path):
        """Load data from pickle file if it exists"""
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Could not load {path}: {e}")
        return None

    def load(self) -> pd.DataFrame:
        """Fetch books systematically for all strategy values"""
        print(" Starting book collection process...\n")
        
        # Initialize value counts for new values
        for strategy in self.SEARCH_STRATEGIES:
            for value in strategy["values"]:
                if (strategy["param"], value) not in self.value_counts:
                    self.value_counts[(strategy["param"], value)] = 0
        
        # Process each search value systematically
        total_processed = 0
        total_values = sum(len(strategy["values"]) for strategy in self.SEARCH_STRATEGIES)
        
        for strategy in tqdm(self.SEARCH_STRATEGIES, desc="Processing strategies"):
            param = strategy["param"]
            for value in tqdm(strategy["values"], desc=f"Processing {param} values"):
                self._fetch_for_value(param, value)
                total_processed += 1
                
                # Show progress every 10 values
                if total_processed % 10 == 0:
                    print(f"\n📊 Progress: {total_processed}/{total_values} values processed")
                    print(f"📚 Books collected so far: {len(self.books_data):,}")
                    print(f"🎯 Target: {total_values * self.MIN_BOOKS_PER_VALUE:,} books")
                
                time.sleep(1)
        
        print(f"\nTotal books collected: {len(self.books_data):,}")
        result_df = pd.DataFrame(self.books_data.values())
        result_df.to_csv("../data/books.csv", index=False)
        return result_df

    def _fetch_for_value(self, param: str, value: str) -> None:
        """Fetch books for a specific parameter value until we have enough"""
        query = f"{param}:{value}"
        start_index = 0
        collected = self.value_counts.get((param, value), 0)  # Resume from previous count
        retry_count = 0
        max_retries = 3
        
        while collected < self.MIN_BOOKS_PER_VALUE:
            try:
                response = requests.get(
                    self.BASE_URL,
                    params={
                        "q": query,
                        "maxResults": self.RESULTS_PER_PAGE,
                        "startIndex": start_index,
                        "key": self.API_KEY,
                        "printType": "books",
                        "langRestrict": "en"
                    },
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()
                
                if "items" not in data:
                    print(f"\nNo more results for '{query}'")
                    break
                
                new_books = self._process_items(data["items"], param, value)
                collected += new_books
                start_index += self.RESULTS_PER_PAGE
                retry_count = 0
                
                if len(data.get("items", [])) < self.RESULTS_PER_PAGE:
                    break
                    
                time.sleep(2)
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"\nMax retries exceeded for '{query}'. Shutting down and saving progress...")
                        self._save_progress()
                        # Save to CSV as well
                        pd.DataFrame(self.books_data.values()).to_csv("../data/books.csv", index=False)
                        print("Progress saved to ../data/books.csv. Exiting.")
                        sys.exit(1)
                    wait_time = min(60 * (2 ** retry_count), 300)
                    print(f"\nRate limited for '{query}'. Waiting {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\nHTTP Error for '{query}': {str(e)}")
                    break
            except requests.exceptions.RequestException as e:
                print(f"\nAPI Error for '{query}': {str(e)}")
                time.sleep(10)
                continue
        
        print(f"\nCollected {collected} books for {param}:{value}")
        # Save progress after each value is processed
        self._save_progress()

    def _process_items(self, items: List[Dict], param: str, value: str) -> int:
        """Process API items and return count of new books added"""
        new_books = 0
        for item in items:
            volume_info = item.get("volumeInfo", {})
            book_id = item.get("id")
            
            # Skip if doesn't meet criteria
            if (not volume_info.get("description") or 
                len(volume_info["description"]) < self.MIN_DESCRIPTION_LENGTH or
                book_id in self.books_data):
                continue
            
            # Store book data
            self.books_data[book_id] = {
                "id": book_id,
                "title": volume_info.get("title", "Untitled"),
                "authors": ", ".join(volume_info.get("authors", ["Unknown"])),
                "description": volume_info.get("description", ""),
                "categories": ", ".join(volume_info.get("categories", [])),
                "primary_category": volume_info.get("categories", ["General"])[0] if volume_info.get("categories") else "General",
                "search_strategy": f"{param}:{value}",
                "retrieval_timestamp": datetime.now().isoformat()
            }
            self.value_counts[(param, value)] += 1
            new_books += 1
        
        return new_books

    def get_coverage_report(self) -> Dict:
        """Generate report showing books collected per search value"""
        report = {}
        for strategy in self.SEARCH_STRATEGIES:
            param = strategy["param"]
            report[param] = {}
            for value in strategy["values"]:
                report[param][value] = self.value_counts.get((param, value), 0)
        return report
    