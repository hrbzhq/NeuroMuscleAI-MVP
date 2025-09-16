# Call for overseas agents and mirror nodes (volunteer recruitment)

To improve access to this project's resources in regions with restricted or slow network access, we are recruiting volunteers globally to host repository mirrors or proxy nodes. Details below:

1) Purpose

- Improve project availability and download speed in different regions.
- Provide alternative access channels for temporary or long-term mirrors (unofficial hosting).

2) Fees and donations

- We do not charge or pay any fees to mirror/proxy hosts — participation is voluntary.
- The project accepts voluntary donations (which will be used for project operating costs and public-interest expenses), but does not require or offer paid relationships with mirrors.

3) Volunteer requirements (optional)

- Ability to host a Git mirror or provide HTTP(S)/rsync access on a server with network connectivity.
- Willingness to cooperate with maintainers in case of security or legal issues (e.g., to take down content if legally required).
- Provide a contact method (email or Telegram/Slack handle) for emergency communication.

4) How to participate (example steps)

1) Create a mirror on the target machine (example):

```bash
git clone --mirror https://github.com/hrbzhq/NeuroMuscleAI-MVP.git repo-mirror.git
# keep the mirror updated periodically
cd repo-mirror.git
git remote update
```

2) Optionally make the mirror available via HTTP/rsync/SCP.

3) Send us your contact information and mirror URL (create an Issue in the repo or email us).

5) Legal & security notes

- Ensure mirrored content complies with local laws and service terms. The maintainers reserve the right to request removal or updates if necessary.

6) Donations

- We accept voluntary donations (for example via PayPal, Open Collective, or other methods). If you want to donate, please open an Issue or email the maintainer to receive donation details.

7) Contact

- To become an agent or mirror host, please open an Issue or email `hrbzhqhrb@gmail.com`.

Thank you for your support and collaboration!
