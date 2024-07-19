# InstaPy Cheatsheet: Building a Marketing Bot for Instagram

## Overview
InstaPy is an open-source tool that allows you to automate interactions on Instagram. It can be used for marketing purposes, such as following users, liking posts, commenting, and engaging with specific hashtags.

## Installation

### Prerequisites
- Python 3.x
- Chrome WebDriver (compatible with your Chrome version)

### Install InstaPy
```bash
pip install instapy
```

## Basic Setup

### Importing InstaPy
```python
from instapy import InstaPy
```

### Starting a Session
```python
session = InstaPy(username='your_username', password='your_password')
session.login()
```

### Setting Up Your Bot

#### Example Configuration
```python
session.set_relationship_bounds(enabled=True, delimit_by_numbers=True, max_followers=5000, min_followers=100)
session.set_do_follow(True, percentage=50)  # Follow 50% of the users you interact with
session.set_do_like(True, percentage=70)     # Like 70% of the posts you interact with
session.set_do_comment(True, percentage=30)   # Comment on 30% of the posts you like
```

## Engaging with Hashtags

### Liking Posts by Hashtag
```python
session.like_by_tags(['#yourhashtag1', '#yourhashtag2'], amount=10)
```

### Commenting on Posts by Hashtag
```python
session.set_comments(['Nice!', 'Great post!', 'Love this!'])
session.comment_by_tags(['#yourhashtag1'], amount=5)
```

## Following Users

### Follow Users from Hashtags
```python
session.follow_by_tags(['#yourhashtag1'], amount=10)
```

### Follow Users from a Specific Profile
```python
session.follow_user_followers(['username'], amount=10, randomize=False)
```

## Unfollowing Users

### Unfollow Users
```python
session.unfollow_users(amount=10, nonFollowers=True)
```

## Ending the Session
```python
session.end()
```

## Complete Example Script

```python
from instapy import InstaPy

# Start a session
session = InstaPy(username='your_username', password='your_password')
session.login()

# Set up bot parameters
session.set_relationship_bounds(enabled=True, delimit_by_numbers=True, max_followers=5000, min_followers=100)
session.set_do_follow(True, percentage=50)
session.set_do_like(True, percentage=70)
session.set_do_comment(True, percentage=30)
session.set_comments(['Nice!', 'Great post!', 'Love this!'])

# Engage with hashtags
session.like_by_tags(['#yourhashtag1', '#yourhashtag2'], amount=10)
session.comment_by_tags(['#yourhashtag1'], amount=5)

# Follow users
session.follow_by_tags(['#yourhashtag1'], amount=10)

# Unfollow users
session.unfollow_users(amount=10, nonFollowers=True)

# End the session
session.end()
```

## Best Practices for Using InstaPy

1. **Avoid Over-Engagement**: Set reasonable limits on likes, follows, and comments to avoid getting your account flagged by Instagram.
2. **Rotate Hashtags**: Use a variety of hashtags to reach different audiences and avoid repetitive engagement.
3. **Monitor Performance**: Keep track of your engagement metrics to adjust your strategy as needed.
4. **Stay Updated**: Regularly update InstaPy and Chrome WebDriver to ensure compatibility with Instagram's latest changes.
5. **Use Proxies**: If you're running multiple accounts or want to avoid IP bans, consider using proxies.

## Conclusion

InstaPy is a powerful tool for automating Instagram marketing tasks. By customizing the bot's behavior, you can effectively engage with your target audience, increase your follower count, and enhance your brand's visibility on the platform. Always follow Instagram's guidelines to maintain a healthy account.

Citations:
[1] https://www.wisegrowthmarketing.com/marketing-best-practices/
[2] https://optinmonster.com/digital-marketing-best-practices/
[3] https://www.youtube.com/watch?v=QH0X4TtBKHs
[4] https://blog.hubspot.com/marketing/digital-strategy-guide
[5] https://www.getcrew.ai
