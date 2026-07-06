from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal,Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage
from pydantic import BaseModel,Field
import operator



# here generator
generator_llm = ChatOpenAI(
    model = "gpt-4.1-mini",
    temperature=0.9
    )
optimizer_llm = ChatOpenAI(
    model = "gpt-4.1-mini",
    temperature=0.8)
evaluator_llm = ChatOpenAI(
    model = "gpt-4.1-mini",
    temperature=0.8)


# Pydantic Schema for structred evaluation output
class TweetEvaluation(BaseModel):
    evaluation : Literal['approved','needs_improvement'] = Field(..., description='final evaluation result of the tweet')
    feedback : str = Field(..., description='feedback for the tweet')

structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluation)


# State
class TweetState(TypedDict):
    topic : str
    tweet : str
    evaluation : Literal["approved","needs_improvement"]
    feedback : str
    iteration : int
    max_iteration : int
    tweet_history : Annotated[list[str],operator.add]



def generator_tweet(state:TweetState):
    #prompt
   messages = [
       SystemMessage(content = "You are funcy and clever Twitter/x influence"),
       HumanMessage(content =f"""
write a short original and hillarious on the topic "{state['topic']}"

Rules:
- Do not use question - answer format
- Max 280 characters
- Use observational humor,irony,sarcasm or cultural refences
- Think in meme logic,puchlines, or relatable takes
- This is version {state['iteration']+1}
""")
   ]
   response = generator_llm.invoke(messages).content
   return {'tweet':response,'tweet_history':[response]}          


def evaluate_tweet(state:TweetState):
   messages = [
      SystemMessage(content = 'You are a ruthless,no-laughs-given twitter critic. You evaluate tweets based on humor,originality,virality and tweet format.'),
      HumanMessage(content = f"""
Evaluate the following tweet:
Tweet:"{state['tweet']}"
Use the criteria below to evaluate the tweet:
1 - Originality : is this fresh,or have you seen it a hundred times before ?
2 - Humor : did it genuinely make you smile,laugh,or chunkle?
3 - Virality potential : would people retweet or share it ?
4 - Format: is it a well - formed tweet(not a setup/punchlines joke,not a question-answer joke, and under 280 character)?

Auto-reject if:
1 - It's written in question-answer format(e.g."why did..." or "what happens when...")
2 - It exceeds 280 characters
3 - It reads like a traditional setup - puchline joke 
4 - It ends with a generic, throwaway, or deflating line that weakens the punch

Respond only in structured format:
evaluation : "approved" or "needs_improvement"
feedback : one paragraph explaining the strengths and weaknesses
""")
    ]
   response = structured_evaluator_llm.invoke(messages)
   return {'evaluation':response.evaluation,'feedback':response.feedback}


def optimize_tweet(state:TweetState):
   messages = [
      SystemMessage(content ="You punch up tweets for virality and humor based on given feedback"),
      HumanMessage(content = f"""
Improve the tweet based on this feedback:
'{state['feedback']}'
Topic:'{state['topic']}'
Original tweet : {state['tweet']}
Re-write it as a short,viral-worthy tweet. Avoid Q&A style. Under 280 characters.
""")
    ]
   response = optimizer_llm.invoke(messages).content
   iteration = state['iteration']+1
   return {'tweet':response,'iteration':iteration,'tweet_history':[response]}



def route_evaluation(state:TweetState):
   if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
      return 'approved'
   else:
      return 'needs_improvement'
   

# Graph
graph = StateGraph(TweetState)

graph.add_node('generate',generator_tweet)
graph.add_node('evaluate',evaluate_tweet)
graph.add_node('optimize',optimize_tweet)


graph.add_edge(START,'generate')
graph.add_edge('generate','evaluate')
graph.add_conditional_edges(
   'evaluate',
   route_evaluation,
   {'approved':END,'needs_improvement':'optimize'}
)

graph.add_edge('optimize','evaluate')

workflow = graph.compile()


initial_state = {
   'topic':"Indian railways",
   'iteration':1,
   'max_iteration':5
}


result = workflow.invoke(initial_state)
print(result)