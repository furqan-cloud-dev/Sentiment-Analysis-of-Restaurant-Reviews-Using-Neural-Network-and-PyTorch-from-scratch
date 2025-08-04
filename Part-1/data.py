# Define Word Dictionaries - Classification by Categories
## n-gram updates ( upto max 3 worlds)

# Using set instead of list for unique values and to further perform intersection for common elements
word_dict = {
    "abusive": {"stupid", "idiot", "dumb", "nonsense"},

    "strongly_negative": {"horrible", "horrible experience", "horrible management", "terrible",
                          "terrible service", "terrible management", "worst", "worst experience",
                          "worst meal", "ridiculous", "never coming back", "never ordering again",
                          "made me sick", "disgusting", "vile", "dirty"},

    "negative": {"bad", "bad experience", "poor", "hate", "hated", "dislike", "not good", "rude",
                 "long wait", "long wait times", "late", "cold food", "incompetent staff",
                 "arrogant staff", "food arrived late", "overpriced", "rude waiters",
                 "noisy environment", "uncomfortable", "uncomfortable seating", "revolting", "wasted",
                 "poor hygiene", "poor hygiene standards", "pathetic service","slow service", "awful service", "unprofessional",
                 "waste of time", "waste of money", "disappointed", "unclean", "unclean tables",
                 "poor presentation", "declined", "would not recommend", "mediocre quality"},

    "neutral": {"okay", "average", "fine", "normal", "menu", "offers", "wide", "everything",
                "tried", "especially", "twists", "classic", "dishes", "food", "kitchen", "service",
                "place", "staff","average", "average food", "ambiance", "waiter","pizza", "experience",
                "seating", "decent prices", "portion", "size", "customers", "customer", "dining",
                "ever", "vegetarian", "nothing special", "parking", "spaces", "flavors", "standard",
                "restaurant", "standard restaurant", "town", "reliable", "fair pricing", "convenient",
                "accessible", "basic", "fine", "satisfactory"},

    "positive": {"good", "good experience", "great", "love", "loved", "like", "nice",
                 "friendly staff", "fresh ingredients", "well-prepared", "well-prepared dishes",
                 "tasty food", "healthy", "healthier", "wonderful", "wonderful flavors",
                 "generous portions", "impressive"},

    "strongly_positive": {"amazing", "amazing experience", "absolutely amazing experience",
                          "amazing food", "amazing attention", "perfect ambiance", "fantastic", "outstanding",
                          "outstanding food", "superb", "excellent", "excellent customer service",
                          "delicious", "perfectly", "perfectly cooked", "great atmosphere",
                          "exceptional service", "incredible flavors", "authentic flavors", "phenomenal", "best",
                          "best restaurant", "highly recommended", "great presentation"},

    "constructive": {"should", "should check", "should check in", "should add", "should add more", "could", "could improve", "could be",
                     "could be more", "could be improved", "could be faster", "could use",
                     "could use more", "arrangement", "suggest", "recommend", "perhaps", "creative",
                     "try", "options", "variety", "more", "often","staff", "attentive",
                     "to customers", "quality", "presentation", "consider", "consider adding",
                     "consider adding more", "consider extending", "consider offering",
                     "online", "online ordering", "ordering", "need", "needs", "needs improvement",
                     "needs attention", "system", "system needs improvement", "delivery", "busy hours", "policies", "wifi connection", "operating hours", "on weekends",
                     "training", "would help", "would be", "would be more", "nice addition", "expensive prices"}
}

