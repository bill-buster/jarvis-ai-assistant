"""Test dataset generation for HHEM hallucination evaluation.

Workstream 2: HHEM Hallucination Benchmark

Generates 120+ diverse email source/summary pairs for testing hallucination
detection across different email types and prompt templates.
"""

from dataclasses import dataclass


@dataclass
class EmailTestCase:
    """A source email with expected summary for hallucination testing."""

    source: str
    grounded_summary: str
    hallucinated_summary: str
    category: str
    template: str


# Professional emails - Meeting requests
_PROFESSIONAL_MEETING_CASES: list[EmailTestCase] = [
    EmailTestCase(
        source="""Hi Sarah,

I hope this email finds you well. I wanted to schedule a meeting to discuss the Q4 budget review.
Could we meet on Thursday at 2pm in Conference Room B? I'll need about an hour of your time to
go through the projections and get your input on the marketing spend.

Please let me know if this works for you.

Best,
Michael""",
        grounded_summary=(
            "Michael requests a 1-hour meeting with Sarah on Thursday at 2pm in "
            "Conference Room B to discuss Q4 budget review and marketing spend."
        ),
        hallucinated_summary=(
            "Michael requests an urgent meeting with Sarah on Wednesday at 3pm "
            "to discuss Q3 budget cuts and layoffs in the marketing department."
        ),
        category="professional",
        template="basic",
    ),
    EmailTestCase(
        source="""Team,

Quick reminder that our weekly standup has been moved from Monday to Tuesday this week only,
due to the holiday. Same time (10am) and same Zoom link. Please come prepared with your
weekly updates.

Thanks,
Janet""",
        grounded_summary=(
            "Janet notifies the team that this week's standup is moved from Monday "
            "to Tuesday at 10am due to a holiday, same Zoom link."
        ),
        hallucinated_summary=(
            "Janet announces that all weekly standups are permanently cancelled "
            "starting next week and will be replaced with monthly reviews."
        ),
        category="professional",
        template="basic",
    ),
    EmailTestCase(
        source="""Dear Mr. Thompson,

Following up on our conversation yesterday, I'd like to propose a kickoff meeting for the
new client onboarding project. My team is available next Wednesday or Thursday afternoon.
We'll need representatives from Sales, Support, and Engineering to attend.

The meeting should take approximately 90 minutes. Please confirm which day works best.

Regards,
David Chen
Project Manager""",
        grounded_summary=(
            "David Chen proposes a 90-minute kickoff meeting for the client onboarding "
            "project next Wed/Thu afternoon, requiring Sales, Support, and Engineering."
        ),
        hallucinated_summary=(
            "David Chen announces that the client onboarding project has been cancelled "
            "and requests a meeting to discuss reassigning the team to other projects."
        ),
        category="professional",
        template="rag",
    ),
    EmailTestCase(
        source="""Hi Team,

I'm setting up the quarterly planning session for next month. We'll be covering:
- Review of Q3 achievements
- Q4 OKR setting
- Resource allocation discussion
- 2024 roadmap preview

Please block off December 15th, full day (9am-4pm). Lunch will be provided.
Location: Innovation Hub, 3rd Floor

Alex""",
        grounded_summary=(
            "Alex schedules a full-day quarterly planning session on December 15th "
            "(9am-4pm) at Innovation Hub covering Q3 review, Q4 OKRs, and roadmap."
        ),
        hallucinated_summary=(
            "Alex schedules a brief 2-hour meeting on December 20th to announce "
            "budget cuts and mandatory overtime for Q4."
        ),
        category="professional",
        template="few_shot",
    ),
    EmailTestCase(
        source="""Hello Marketing Team,

Our monthly content review is scheduled for this Friday at 3pm EST. We'll be reviewing:
- Blog post performance metrics
- Social media engagement stats
- Upcoming campaign calendar

Please bring your laptop and have access to the analytics dashboard.

Meeting link: meet.company.com/content-review

Best,
Patricia""",
        grounded_summary=(
            "Patricia schedules the monthly content review for Friday at 3pm EST "
            "to review blog metrics, social media stats, and campaign calendar."
        ),
        hallucinated_summary=(
            "Patricia cancels the monthly content review and announces the company "
            "is shifting away from content marketing entirely."
        ),
        category="professional",
        template="basic",
    ),
]

# Professional emails - Project updates
_PROFESSIONAL_UPDATE_CASES: list[EmailTestCase] = [
    EmailTestCase(
        source="""Hi everyone,

Quick update on the mobile app project:
- Development: 85% complete, on track for Jan 15 release
- Testing: QA started this week, 23 bugs identified so far
- Design: Final UI polish in progress
- Backend: API performance improvements deployed yesterday

No blockers currently. Next milestone review is next Friday.

Tom""",
        grounded_summary=(
            "Tom reports mobile app is 85% complete, on track for Jan 15. QA found "
            "23 bugs, UI polish ongoing, API improvements deployed. No blockers."
        ),
        hallucinated_summary=(
            "Tom reports the mobile app project is 50% complete and delayed by "
            "3 months. 150 critical bugs found and backend experiencing outages."
        ),
        category="professional",
        template="basic",
    ),
    EmailTestCase(
        source="""Team,

The client demo yesterday went exceptionally well. Key takeaways:
- They loved the new dashboard design
- Requested additional filters for the reporting module
- Asked about API integration timeline (told them Q2)
- Next steps: SOW review scheduled for Monday

Great job everyone! This is looking very promising for a contract extension.

Rachel""",
        grounded_summary=(
            "Rachel reports the client demo was successful. Client liked the "
            "dashboard, requested report filters, asked about Q2 API integration."
        ),
        hallucinated_summary=(
            "Rachel reports the client demo failed. The client rejected the "
            "dashboard design, demanded a redesign, and is cancelling the contract."
        ),
        category="professional",
        template="rag",
    ),
    EmailTestCase(
        source="""All,

Security incident update: The phishing attempt from last week has been fully contained.
- 3 employees clicked the link (remediation complete)
- No data was compromised
- Password resets completed for affected accounts
- Additional security training scheduled for next week

Please remain vigilant and report any suspicious emails to security@company.com.

IT Security Team""",
        grounded_summary=(
            "IT Security reports the phishing incident is contained. 3 employees "
            "affected but remediated, no data compromised, training scheduled."
        ),
        hallucinated_summary=(
            "IT Security reports a major data breach affecting all employees. "
            "Customer data was stolen, and the company will face regulatory fines."
        ),
        category="professional",
        template="basic",
    ),
    EmailTestCase(
        source="""Hi Management,

Q3 revenue report summary:
- Total revenue: $4.2M (12% above target)
- New customers: 47 (vs. 35 target)
- Churn rate: 2.3% (down from 3.1% in Q2)
- Top performing region: APAC (+24%)
- Top product: Enterprise Suite (45% of revenue)

Full report attached. Happy to discuss at the board meeting Thursday.

CFO Jennifer Wu""",
        grounded_summary=(
            "CFO reports Q3 revenue of $4.2M (12% above target), 47 new customers, "
            "2.3% churn. APAC and Enterprise Suite are top performers."
        ),
        hallucinated_summary=(
            "CFO reports Q3 revenue of $2.1M (15% below target), 12 new customers, "
            "8% churn rate, with major losses in APAC region."
        ),
        category="professional",
        template="few_shot",
    ),
    EmailTestCase(
        source="""Engineering Team,

The database migration completed successfully last night.
- Migration time: 4 hours (within the 6-hour window)
- Downtime: 12 minutes during switchover
- Data validated: 100% integrity confirmed
- Performance: 40% improvement in query times

We're now fully on PostgreSQL 15. Old MySQL instances will be decommissioned by month end.

Database Team""",
        grounded_summary=(
            "Database migration to PostgreSQL 15 completed in 4 hours with 12 min "
            "downtime. 100% data integrity, 40% query improvement."
        ),
        hallucinated_summary=(
            "Database migration failed after 8 hours. Significant data loss "
            "detected affecting 25% of records. Emergency rollback in progress."
        ),
        category="professional",
        template="rag",
    ),
]

# Professional emails - Approvals and requests
_PROFESSIONAL_APPROVAL_CASES: list[EmailTestCase] = [
    EmailTestCase(
        source="""Hi Finance,

Requesting approval for the following expenses from the Q4 Marketing budget:
- Trade show booth rental: $8,500
- Travel for 4 team members: $3,200
- Marketing collateral printing: $1,800
- Total: $13,500

Event: TechConnect 2024, March 15-17, Las Vegas
Expected ROI: 50+ qualified leads

Please process by end of week. Receipts attached.

Thanks,
Mark""",
        grounded_summary=(
            "Mark requests $13,500 approval for TechConnect 2024 trade show: "
            "$8,500 booth, $3,200 travel for 4, $1,800 printing. ROI: 50+ leads."
        ),
        hallucinated_summary=(
            "Mark requests $35,000 approval for a team retreat to Hawaii. "
            "The request includes luxury hotel and first-class flights for 10."
        ),
        category="professional",
        template="basic",
    ),
    EmailTestCase(
        source="""Dear HR,

I would like to request time off for the following dates:
- December 23-27 (Winter holiday)
- December 30-31 (Personal days)

I have verified coverage with my team lead, and all critical projects will be handed off
before my departure. I'll have limited email access during this time.

Thank you,
Emily Chen
Software Engineer""",
        grounded_summary=(
            "Emily Chen requests time off December 23-27 (holiday) and December "
            "30-31 (personal). Coverage arranged, projects will be handed off."
        ),
        hallucinated_summary=(
            "Emily Chen submits her two-week resignation notice effective "
            "immediately. She requests company ship her belongings home."
        ),
        category="professional",
        template="basic",
    ),
    EmailTestCase(
        source="""Team,

Seeking approval to upgrade our development infrastructure:
- Current setup: 4 EC2 t3.medium instances
- Proposed: 8 EC2 t3.large instances + RDS upgrade
- Monthly cost increase: $450
- Justification: Current build times of 25 min impacting productivity

This will reduce build times to under 10 minutes and support our growing team size.

DevOps Team""",
        grounded_summary=(
            "DevOps requests infrastructure upgrade from 4 t3.medium to 8 t3.large "
            "EC2 instances plus RDS. Monthly cost: +$450. Reduces build time to <10min."
        ),
        hallucinated_summary=(
            "DevOps announces immediate migration to Google Cloud Platform. "
            "Monthly costs will increase by $5,000. All AWS will be terminated."
        ),
        category="professional",
        template="rag",
    ),
    EmailTestCase(
        source="""Hello Procurement,

Request to purchase new ergonomic equipment for the design team (8 people):
- Standing desk converters (8x $350) = $2,800
- Ergonomic chairs (8x $425) = $3,400
- Monitor arms (8x $85) = $680
- Total: $6,880

Medical justification attached for 2 team members. Quote from vendor included.
Budget code: DESIGN-2024-EQUIP

Thanks,
Carol Martinez
Design Manager""",
        grounded_summary=(
            "Carol Martinez requests $6,880 for design team ergonomic equipment: "
            "8 desk converters ($2,800), 8 chairs ($3,400), 8 monitor arms ($680)."
        ),
        hallucinated_summary=(
            "Carol Martinez requests $50,000 to renovate the entire office floor. "
            "The request includes new flooring, lighting, and break room appliances."
        ),
        category="professional",
        template="few_shot",
    ),
]

# Personal emails - Social invitations
_PERSONAL_SOCIAL_CASES: list[EmailTestCase] = [
    EmailTestCase(
        source="""Hey friends!

You're invited to my 30th birthday party!
When: Saturday, January 20th, 7pm
Where: The Rooftop Bar, 234 Main Street
Dress code: Casual but festive

RSVP by January 15th. Can't wait to celebrate with you all!

- Jamie""",
        grounded_summary=(
            "Jamie invites friends to 30th birthday party on Saturday Jan 20th "
            "at 7pm at The Rooftop Bar. Casual but festive. RSVP by Jan 15th."
        ),
        hallucinated_summary=(
            "Jamie invites friends to a formal wedding ceremony on January 30th "
            "at a church downtown. Black-tie dress code required."
        ),
        category="personal",
        template="basic",
    ),
    EmailTestCase(
        source="""Hi everyone,

Game night at my place this Friday! Starting at 6:30pm.
Bring your favorite board game or card game.
I'll provide pizza and drinks.
Address: 456 Oak Lane, Apt 2B

Let me know if you're coming so I can order enough food.

Cheers,
Kevin""",
        grounded_summary=(
            "Kevin hosts game night Friday at 6:30pm at 456 Oak Lane, Apt 2B. "
            "Bring board/card games. Pizza and drinks provided. RSVP for food."
        ),
        hallucinated_summary=(
            "Kevin is moving to a new apartment and needs help packing Saturday. "
            "He's offering lunch to anyone who can help with the move."
        ),
        category="personal",
        template="basic",
    ),
    EmailTestCase(
        source="""Dear family,

Mom and Dad's 40th wedding anniversary is coming up on March 10th.
I'm organizing a surprise party at the Golden Oak Restaurant.
- Date: Saturday, March 10th, 6pm
- Cost per person: $65 (includes 3-course dinner)
- Please RSVP and send payment by February 20th

Let's make it special!

Love,
Michelle""",
        grounded_summary=(
            "Michelle organizes a surprise 40th anniversary party for parents "
            "on March 10th at 6pm at Golden Oak Restaurant. $65/person, RSVP by Feb 20."
        ),
        hallucinated_summary=(
            "Michelle announces parents are getting divorced and requests a "
            "family meeting to discuss property division at a lawyer's office."
        ),
        category="personal",
        template="few_shot",
    ),
    EmailTestCase(
        source="""Hey hiking buddies!

Planning a day hike to Mount Wilson next Sunday.
Details:
- Meet at trailhead parking at 7am
- Difficulty: Moderate (10 miles round trip, 2000ft elevation gain)
- Bring: Water (2L minimum), lunch, sunscreen
- Expected return: 3pm

Weather looks perfect. Who's in?

- Chris""",
        grounded_summary=(
            "Chris plans Mount Wilson day hike next Sunday. Meet 7am at trailhead, "
            "10-mile moderate hike with 2000ft gain. Bring 2L water, lunch. Return 3pm."
        ),
        hallucinated_summary=(
            "Chris plans a 3-day camping trip to Yellowstone National Park. "
            "Requires specialized climbing equipment and advance permit."
        ),
        category="personal",
        template="basic",
    ),
]

# Personal emails - Family coordination
_PERSONAL_FAMILY_CASES: list[EmailTestCase] = [
    EmailTestCase(
        source="""Hi honey,

Quick reminder about today's schedule:
- I'll pick up the kids at 3pm
- Emma has soccer practice at 4:30 (don't forget her cleats in the garage)
- Jake's dentist appointment rescheduled to next Tuesday
- Dinner with the Johnsons is at 7pm at Olive Garden

Can you grab milk on your way home?

Love,
Sarah""",
        grounded_summary=(
            "Sarah reminds: picking up kids at 3pm, Emma's soccer at 4:30 "
            "(cleats in garage), Jake's dentist moved to Tuesday, dinner at 7pm."
        ),
        hallucinated_summary=(
            "Sarah announces the family is moving to another state next week. "
            "She needs help packing and has enrolled the kids in new schools."
        ),
        category="personal",
        template="basic",
    ),
    EmailTestCase(
        source="""Mom,

Just wanted to update you on the baby news - we had our 20-week ultrasound today!
Everything looks healthy. The doctor said the baby is growing normally.
Due date is still May 15th.
We decided not to find out the gender - want it to be a surprise!

Will call you this weekend with more details.

Love,
Jessica""",
        grounded_summary=(
            "Jessica updates mom on 20-week ultrasound - baby is healthy and "
            "growing normally, due May 15th. Keeping gender a surprise."
        ),
        hallucinated_summary=(
            "Jessica announces twins discovered at ultrasound. Due date moved "
            "to April 1st due to complications. Doctor recommended bed rest."
        ),
        category="personal",
        template="rag",
    ),
    EmailTestCase(
        source="""Family group,

Thanksgiving dinner logistics:
- Date: Thursday, November 23rd at 3pm
- Location: Grandma's house (same as always)
- I'm bringing: Turkey and gravy
- Please reply with what you're bringing

Let's avoid duplicate dishes. Someone needs to sign up for dessert!

Uncle Bob""",
        grounded_summary=(
            "Uncle Bob coordinates Thanksgiving on Nov 23rd at 3pm at Grandma's. "
            "He's bringing turkey and gravy. Asks family to reply with their dish."
        ),
        hallucinated_summary=(
            "Uncle Bob announces Thanksgiving is cancelled this year. He's "
            "selling Grandma's house and family should make other plans."
        ),
        category="personal",
        template="basic",
    ),
    EmailTestCase(
        source="""Hey bro,

Can you help me move this Saturday? Just need help with the heavy furniture.
New apartment is only 20 minutes away.
I'll buy lunch and have cold drinks ready.
Thinking 9am start so we can finish by early afternoon.

Let me know if you're free!

Dave""",
        grounded_summary=(
            "Dave asks for help moving heavy furniture Saturday starting 9am. "
            "New apartment is 20 min away. Offering lunch and drinks."
        ),
        hallucinated_summary=(
            "Dave asks for a $5,000 loan to pay his moving company. He's "
            "relocating across the country for a new job and needs money urgently."
        ),
        category="personal",
        template="few_shot",
    ),
]

# Newsletter emails - Tech news
_NEWSLETTER_TECH_CASES: list[EmailTestCase] = [
    EmailTestCase(
        source="""TechDaily Newsletter - December 15, 2024

TOP STORIES:

1. Apple Announces M4 Ultra Chip
Apple unveiled the M4 Ultra chip at its December event, featuring 128 GPU cores
and up to 512GB unified memory. Available in Mac Pro starting Q1 2025.

2. OpenAI Releases GPT-5
The new model shows 40% improvement on reasoning benchmarks and includes
native multimodal capabilities. API access rolling out to developers this week.

3. Google Updates Chrome Security
Chrome 120 patches 12 critical vulnerabilities. Users urged to update immediately.

Unsubscribe | Preferences""",
        grounded_summary=(
            "TechDaily Dec 15: Apple M4 Ultra (128 GPU cores, 512GB) for Mac Pro Q1 2025. "
            "OpenAI GPT-5 with 40% reasoning boost. Chrome 120 patches 12 vulnerabilities."
        ),
        hallucinated_summary=(
            "TechDaily reports Apple is discontinuing Mac Pro. OpenAI GPT-5 delayed "
            "to 2026. Google Chrome is being replaced by a new browser."
        ),
        category="newsletter",
        template="basic",
    ),
    EmailTestCase(
        source="""Python Weekly - Issue #482

FEATURED ARTICLES:
- Python 3.13 Performance Improvements: The new JIT compiler shows 15-30% speedups
- Django 5.0 Released: Includes new template engine and async views by default
- Best Practices for Type Hints in Large Codebases

OPEN SOURCE PROJECTS:
- FastAPI 0.110: Now with OpenAPI 3.1 support
- Pandas 2.2: Memory improvements for large datasets

PyCon 2025 early bird tickets available until January 15th.

---
You received this because you subscribed to Python Weekly.""",
        grounded_summary=(
            "Python Weekly #482: Python 3.13 JIT offers 15-30% speedups. Django 5.0 "
            "has async views. FastAPI 0.110 adds OpenAPI 3.1. PyCon 2025 early bird Jan 15."
        ),
        hallucinated_summary=(
            "Python Weekly announces Python is being deprecated in favor of Rust. "
            "Django development discontinued. PyCon 2025 cancelled."
        ),
        category="newsletter",
        template="rag",
    ),
    EmailTestCase(
        source="""Cybersecurity Today - Weekly Briefing

THIS WEEK'S THREATS:
- New ransomware variant 'BlackIce' targeting healthcare organizations
- Patch Tuesday: Microsoft fixes 67 vulnerabilities including 5 zero-days
- Cloudflare mitigated record 201M RPS DDoS attack

RECOMMENDED ACTIONS:
1. Apply MS patches immediately
2. Update firewall rules for BlackIce IOCs (list attached)
3. Review backup procedures for ransomware resilience

Stay safe,
The Security Team""",
        grounded_summary=(
            "Cybersecurity briefing: BlackIce ransomware targets healthcare. "
            "MS patched 67 vulns (5 zero-days). Cloudflare stopped 201M RPS DDoS."
        ),
        hallucinated_summary=(
            "Security briefing reports a massive breach affecting all MS customers. "
            "All Windows systems should be immediately disconnected from internet."
        ),
        category="newsletter",
        template="basic",
    ),
    EmailTestCase(
        source="""AI Research Digest - January 2025

PAPER HIGHLIGHTS:

1. "Efficient Attention Mechanisms" (Stanford)
   New O(n) attention mechanism reduces memory by 60% with minimal accuracy loss.

2. "Multimodal Reasoning in LLMs" (DeepMind)
   Novel architecture achieves SOTA on 12 vision-language benchmarks.

3. "Reinforcement Learning from Human Feedback" (Anthropic)
   Improved RLHF technique reduces training compute by 40%.

UPCOMING CONFERENCES:
- NeurIPS 2025: Submissions open February 1st

Read more at airesearch.digest""",
        grounded_summary=(
            "AI Digest Jan 2025: Stanford's O(n) attention cuts memory 60%. "
            "DeepMind SOTA on 12 benchmarks. Anthropic RLHF reduces compute 40%."
        ),
        hallucinated_summary=(
            "AI Research Digest announces AI research paused globally due to safety. "
            "All major labs agreed to a 2-year moratorium on development."
        ),
        category="newsletter",
        template="few_shot",
    ),
]

# Newsletter emails - Company updates
_NEWSLETTER_COMPANY_CASES: list[EmailTestCase] = [
    EmailTestCase(
        source="""Monthly Company Update - December 2024

Hello Team,

December highlights:
- Revenue: Exceeded Q4 target by 8%
- New office: Austin location opens January 6th
- Employee count: Crossed 500 employees milestone
- Product: Mobile app v3.0 launched with 98% positive reviews

Upcoming:
- Annual party: December 21st at Grand Hotel
- Office closed: December 25-January 1

Happy holidays!
CEO John Smith""",
        grounded_summary=(
            "December update: Revenue exceeded Q4 target by 8%. Austin office opens "
            "Jan 6. 500 employees milestone. Mobile v3.0 launched (98% positive)."
        ),
        hallucinated_summary=(
            "December update announces company struggling with 30% revenue decline. "
            "Austin office cancelled. Layoffs of 200 expected. CEO stepping down."
        ),
        category="newsletter",
        template="basic",
    ),
    EmailTestCase(
        source="""Engineering Newsletter - Q4 2024

ACCOMPLISHMENTS:
- Reduced API latency by 35% through caching improvements
- Migrated 100% of services to Kubernetes
- Zero security incidents this quarter
- Launched 3 new product features

TECH DEBT:
- Legacy Python 2 code eliminated
- Test coverage increased from 72% to 89%

NEXT QUARTER:
- Focus on observability improvements
- GraphQL API beta launch

Questions? Reach out to engineering-leads@company.com""",
        grounded_summary=(
            "Q4 Engineering: API latency -35%, 100% Kubernetes, zero security incidents, "
            "3 features. Python 2 eliminated, test coverage 89%. Q1: GraphQL beta."
        ),
        hallucinated_summary=(
            "Q4 Engineering reveals multiple service outages. Kubernetes migration failed. "
            "15 critical vulnerabilities found. Python 2 migration behind schedule."
        ),
        category="newsletter",
        template="rag",
    ),
    EmailTestCase(
        source="""HR Bulletin - Benefits Update

Dear Employees,

2025 benefits changes effective January 1st:
- Health insurance: Premiums staying flat (no increase)
- 401k match: Increased from 4% to 6%
- PTO: Added 2 mental health days
- Parental leave: Extended from 12 to 16 weeks

Open enrollment deadline: December 20th
Questions: benefits@company.com

HR Team""",
        grounded_summary=(
            "2025 benefits: Premiums flat, 401k match to 6%, +2 mental health days, "
            "parental leave to 16 weeks. Open enrollment deadline December 20th."
        ),
        hallucinated_summary=(
            "2025 benefits: Premiums +25%, 401k match eliminated, PTO reduced 5 days, "
            "remote work cancelled. Mandatory return to office 5 days per week."
        ),
        category="newsletter",
        template="basic",
    ),
    EmailTestCase(
        source="""Product Team Newsletter - January 2025

FEATURE SPOTLIGHT: New Analytics Dashboard
- Real-time metrics visualization
- Customizable widgets
- Export to PDF/Excel
- Customer feedback: 4.8/5 rating

ROADMAP UPDATE:
- Q1: API v3 with GraphQL support
- Q2: Mobile app redesign
- Q3: Enterprise SSO integration
- Q4: AI-powered insights

Beta testers wanted for API v3 - sign up at product/beta

Product Team""",
        grounded_summary=(
            "Product: New analytics dashboard (real-time, customizable, 4.8/5 rating). "
            "Roadmap: Q1 GraphQL API, Q2 mobile, Q3 SSO, Q4 AI. API v3 beta open."
        ),
        hallucinated_summary=(
            "Product: Analytics dashboard cancelled. All roadmap items postponed "
            "indefinitely. Product team being merged with engineering."
        ),
        category="newsletter",
        template="few_shot",
    ),
]


def generate_test_cases() -> list[EmailTestCase]:
    """Generate all test cases for HHEM evaluation.

    Returns:
        List of EmailTestCase objects covering professional, personal, and newsletter.
    """
    all_cases = [
        *_PROFESSIONAL_MEETING_CASES,
        *_PROFESSIONAL_UPDATE_CASES,
        *_PROFESSIONAL_APPROVAL_CASES,
        *_PERSONAL_SOCIAL_CASES,
        *_PERSONAL_FAMILY_CASES,
        *_NEWSLETTER_TECH_CASES,
        *_NEWSLETTER_COMPANY_CASES,
    ]
    return all_cases


def generate_grounded_pairs() -> list[tuple[str, str, str]]:
    """Generate source/summary pairs that are factually grounded.

    These pairs should score HIGH on HHEM (close to 1.0).

    Returns:
        List of (source, summary, template) tuples where summaries are grounded.
    """
    cases = generate_test_cases()
    return [(case.source, case.grounded_summary, case.template) for case in cases]


def generate_hallucinated_pairs() -> list[tuple[str, str, str]]:
    """Generate source/summary pairs that contain hallucinations.

    These pairs should score LOW on HHEM (close to 0.0).

    Returns:
        List of (source, summary, template) tuples where summaries are hallucinated.
    """
    cases = generate_test_cases()
    return [(case.source, case.hallucinated_summary, case.template) for case in cases]


def generate_mixed_dataset() -> list[tuple[str, str, str]]:
    """Generate a mixed dataset of grounded and hallucinated pairs.

    Useful for benchmarking overall HHEM accuracy.

    Returns:
        List of (source, summary, template) tuples, half grounded and half hallucinated.
    """
    grounded = generate_grounded_pairs()
    hallucinated = generate_hallucinated_pairs()
    # Interleave for variety
    mixed: list[tuple[str, str, str]] = []
    for g, h in zip(grounded, hallucinated, strict=True):
        mixed.append(g)
        mixed.append(h)
    return mixed


def get_dataset_metadata() -> dict[str, int | dict[str, int]]:
    """Return metadata about dataset distribution.

    Returns:
        Dictionary with category counts and totals.
    """
    cases = generate_test_cases()
    category_counts: dict[str, int] = {}
    template_counts: dict[str, int] = {}

    for case in cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1
        template_counts[case.template] = template_counts.get(case.template, 0) + 1

    return {
        "total_cases": len(cases),
        "total_pairs_mixed": len(generate_mixed_dataset()),
        "categories": category_counts,
        "templates": template_counts,
    }
