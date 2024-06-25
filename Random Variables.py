import matplotlib.pyplot as plt
import random
import math
def line_space(lower_bound, upper_bound,npoints):#generates npoints no. of boints within a given interval.
    division = (upper_bound-lower_bound)/npoints # calculates the difference between each point
    points  = []
    current_point = lower_bound
    for _ in range(npoints+1): #adds division to each point starting from the lower bound and appends them to the points list.
        points.append(current_point)
        current_point += division
    return points
def generate_bernoulli_randvar(n,p):
    return random.choices([0,1],weights=[(1-p),p],k=n) #chooses between the list of input with "weights" as the probabikity of each list entery respectively.
def binomial_pmf(n,r,p):
    return math.comb(n,r)*p**r*(1-p)**(n-r)
def binomial(n, p,size):
    pmf = [binomial_pmf(n,i,p) for i in range(n+1)] #calculate the pmf of 0 to n numbers
    x = [i for i in range(n+1)] # list of random variable values from 0 to n
    cdf = [sum(pmf[:i+1])for i in range(n+1)] # calculates the cdf at each point
    # generates the random variables which are the no. of successes in n bernoulli experiments over size no. of samples.
    histogram = [generate_bernoulli_randvar(n,p).count(1) for _ in range(size)]
    mean = n*p
    variance = n*p*(1-p)
    #plots the pmf, cdf and histogram graphs.
    plt.vlines(x, 0, pmf, lw=20, alpha=1)
    plt.text(0, 0.27, f'E: {mean}\nvar: {variance}', fontsize=12)
    plt.title("binomial PMF")
    plt.xlabel("X")
    plt.ylabel("probability")
    plt.show()
    plt.step(x,cdf, lw=5,where="post")
    plt.title("binomial CDF")
    plt.xlabel("X")
    plt.ylabel("cummulative probability")
    plt.show()
    plt.hist(histogram,bins=n+1,align='mid',edgecolor='black')
    plt.title("binomial histogram")
    plt.xlabel("X")
    plt.ylabel("samples")
    plt.show()
def bernoulli_pmf(probability,number):
    if number == 1:
        return probability
    elif number == 0:
        return 1-probability
def bernoulli_cdf(probability,number):
    if number == 1:
        return 1
    elif number == 0:
        return 1-probability
def bernoulli(probability, size):
    x = [0,1] #failure or success.
    pmf = [bernoulli_pmf(probability, i) for i in x] #probability of 0 and1
    cdf = [bernoulli_cdf(probability, i) for i in x]# cummulative probability of 0 and 1
    histogram = generate_bernoulli_randvar(size, probability)# generates size no. of bernoulli random variables.
    mean = probability
    variance = probability * (1 - probability)
    #plots the pmf, cdf and histogram graphs.
    plt.vlines(x, 0, pmf, lw=40, alpha=1)
    plt.text(0.5, 0.05, f'E: {mean}\nvar: {variance}', fontsize=12)
    plt.title("bernoulli PMF")
    plt.xlabel("X")
    plt.ylabel("probability")
    plt.show()
    plt.step(x,cdf,lw=10,where='post')
    plt.title("bernoulli CDF")
    plt.xlabel("X")
    plt.ylabel("cummulative probability")
    plt.show()
    plt.hist(histogram, bins='auto')
    plt.title("bernoulli histogram")
    plt.xlabel("X")
    plt.ylabel("samples")
    plt.show()
def geometric_pmf(probability,numer_of_trials):
        return ((1-probability)**(numer_of_trials-1))*probability
def geometric_cdf(probability,no_of_trials):
    return 1-((1-probability)**no_of_trials)
def genetrate_geometric_randvar(probability):
    count = 0
    while True:
        result = generate_bernoulli_randvar(1, probability)
        count = count + 1
        if result == [1]:
            break
    return count
def geometric(probability, size):
    x = [i for i in range(1,30)] #sample of random variables limit 20.
    pmf = [geometric_pmf(probability, i) for i in x] # calculates pmf values for each element in x.
    cdf = [geometric_cdf(probability, i) for i in x] # calculates cdf values for each element in x.
    histogram = [genetrate_geometric_randvar(probability) for _ in range(size)] #generates size no. of random variables
    mean = 1 / probability
    variance = (1 - probability) / (probability ** 2)
    #plots the pmf, cdf and histogram graphs.
    plt.vlines(x, 0, pmf, lw=20, alpha=1)
    plt.text(7, 0.05, f'E: {mean}\nvar: {variance}', fontsize=12)
    plt.title("geometric PMF")
    plt.xlabel("X")
    plt.ylabel("probability")
    plt.show()
    plt.step(x,cdf, lw=3,where='post')
    plt.title("geometric CDF")
    plt.xlabel("X")
    plt.ylabel("cummulative probability")
    plt.show()
    plt.hist(histogram, bins=max(histogram)+1,align="mid",edgecolor='black')
    plt.title("geometric histogram")
    plt.xlabel("X")
    plt.ylabel("samples")
    plt.show()
def poisson_pmf(lambda_,k):
    return (math.exp(-lambda_) * (lambda_**k) / math.factorial(k))
def poisson_cdf(lambda_, k):
    # calculate the pmf values up to k and sum them.
    return sum(poisson_pmf(lambda_,i) for i in range(k+1))
def inverse_transform_sampling_poisson(lambda_):
    u = random.uniform(0, 1)
    k = 0
    cumulative_prob = poisson_cdf(lambda_, k)
    while u > cumulative_prob:
        k += 1
        cumulative_prob += poisson_pmf(lambda_, k)
    return k
def generate_poisson_randvar(lambda_,size):
    return [inverse_transform_sampling_poisson(lambda_) for _ in range(size)]
def poisson(lambda_,lower_bound,upper_bound,size):
    #create list of x values max 20.
    x = [i for i in range(lower_bound,upper_bound+1)]
    #calculate the pmf value for each x value.
    pmf = [poisson_pmf(lambda_,i) for i in x]
    #calculate the cdf at each x value.
    cdf = [poisson_cdf(lambda_,i) for i in x]
    #calculate mean and variance.
    mean = variance = lambda_
    #generate the random variables for the histogram.
    histogram = generate_poisson_randvar(lambda_,size)
    #plots the pmf, cdf and histogram graphs.
    plt.vlines(x, 0, pmf, lw=12)
    plt.text(10, 0.1, f'E: {mean}\nvar: {variance}', fontsize=12)
    plt.title("poisson PMF")
    plt.xlabel("X")
    plt.ylabel("probability")
    plt.show()
    plt.step(x,cdf, lw=5,where="post")
    plt.title("poisson CDF")
    plt.xlabel("X")
    plt.ylabel("probability")
    plt.show()
    plt.hist(histogram, bins=max(histogram)+1,edgecolor='black')
    plt.title("poisson histogram")
    plt.xlabel("X")
    plt.ylabel("samples")
    plt.show()
def discrete_uniform_pmf(lower_bound,upper_bound , point):
    if upper_bound>=point>=lower_bound:
        return 1/(upper_bound-lower_bound+1)
    else:
        return 0
def discrete_uniform_cdf(lower_bound,upper_bound,point):
    if lower_bound <= point <= upper_bound:
        return (point - lower_bound+1) / (upper_bound - lower_bound+1)
    elif point > upper_bound:
        return 1
    else:
        return 0
def generate_uniform_randvar(a,b,size):
    #chooses a piont randomly size number of times and store it in a list
    return [random.randrange(a,b+1) for _ in range(size)]
def uniform_discrete(a,b,size):
    x = [i for i in range(a,b+1)] #each point within the given interval
    # sets the probability and duplicate it for the sixe of the interval.
    pmf = [discrete_uniform_pmf(a,b,i) for i in x]
    # since all the pmf values are equal all
    # we need is to multiply the probability by itt's index +1 ie:1,2,3...
    cdf = [discrete_uniform_cdf(a,b,i) for i in x]
    # generates all the random variables needed for the histogram.
    histogram = generate_uniform_randvar(a,b,size)
    mean = (b-a)/2
    variance = round((((b-a+1)**2)-1)/12,1)
    # calculates pmf values for each element in x.
    plt.vlines(x, 0, pmf, lw=20, alpha=1)
    plt.text(a, 1/(b-a), f'E: {mean}\nvar: {variance}', fontsize=12)
    plt.title("uniform PMF")
    plt.xlabel("n")
    plt.ylabel("probability")
    plt.show()
    plt.step(x,cdf, lw=5)
    plt.title("uniform CDF")
    plt.xlabel("X")
    plt.ylabel("probability")
    plt.show()
    plt.hist(histogram, bins=len(x),edgecolor='black',align='mid')
    plt.xticks(x)
    plt.title("uniform histogram")
    plt.xlabel("X")
    plt.ylabel("samples")
    plt.show()
def continous_uniform_pmf(lower_bound, upper_bound, point):
    if upper_bound>=point>=lower_bound:
        return 1/(upper_bound-lower_bound)
    else:
        return 0
def continous_uniform_cdf(lower_bound,upper_bound,point):
    if lower_bound <= point <= upper_bound:
        return (point - lower_bound) / (upper_bound - lower_bound)
    elif point > upper_bound:
        return 1
    else:
        return 0
def generate_continous_uniform_randvar(upper_bound,lower_bound,size):
    return [random.uniform(lower_bound,upper_bound) for _ in range(size)]
def continous_uniform(lower_bound,upper_bound,npoints,size):
    # gets the points to calculate the distribution with.
    x = line_space(lower_bound-10,upper_bound+10,npoints)
    #generates the random variables for histogram.
    histogram = generate_continous_uniform_randvar(upper_bound,lower_bound,size)
    # calculates the pmf value for each x value.
    pdf = [continous_uniform_pmf(lower_bound, upper_bound, i) for i in x]
    # calculates the pdf value for each x value.
    cdf = [continous_uniform_cdf(lower_bound,upper_bound,i) for i in x]
    mean = (upper_bound - lower_bound) / 2
    variance = round((((upper_bound - lower_bound + 1) ** 2) - 1) / 12, 1)
    #plots the PDF, CDF and histogram.
    plt.plot(x, pdf, label='PDF')
    plt.text(lower_bound, (1 / (upper_bound - lower_bound))/2, f'E: {mean}\nvar: {variance}', fontsize=12)
    plt.title('Continuous Uniform PDF')
    plt.xlabel('Value')
    plt.ylabel('PDF')
    plt.legend()
    plt.show()
    plt.plot(x, cdf, label='CDF')
    plt.title('Continuous Uniform CDF')
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()
    plt.hist(histogram, bins='auto', edgecolor='black',align='mid')
    plt.title("uniform histogram")
    plt.xlabel("samples")
    plt.ylabel("probability")
    plt.show()
def generate_exponential_random_variables(lambda_, size=1000):
    return [(-1 / lambda_) * math.log(random.uniform(0,1)) for _ in range(size)]
def exponential_pdf(lambda_,point):
    return lambda_*math.exp(-lambda_*point) # returns the pdf at point
def exponential_cdf(lambda_,point):
    return 1-math.exp(-lambda_*point) # returns the cdf at point
def exponential(lambda_,lower_bound,upper_bound,npoints,size):
    points = line_space(lower_bound,upper_bound,npoints) # creates the interval of values of the random variables
    histogram = generate_exponential_random_variables(lambda_,size) # generates the random variables for the histogram.
    ypdf = [exponential_pdf(lambda_,x) for x in points] #calculates the pdf for each value of x
    ycdf = [exponential_cdf(lambda_,x) for x in points] # calculates the cdf for each value of x
    mean = 1/lambda_
    variance = 1/lambda_**2
    # plots the PDF, CDF and histogram.
    plt.plot(points, ypdf, label=f'λ = {lambda_}')
    plt.text(lower_bound, 1 / (upper_bound - lower_bound), f'E: {mean}\nvar: {variance}', fontsize=12)
    plt.title('Exponential PDF')
    plt.xlabel('Value')
    plt.ylabel('PDF')
    plt.legend()
    plt.show()
    plt.plot(points, ycdf, label=f'λ = {lambda_}')
    plt.title('Exponential CDF')
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()
    plt.hist(histogram, bins=round(max(histogram)), edgecolor='black')
    plt.title("Exponential histogram")
    plt.xlabel("X")
    plt.ylabel("probability")
    plt.show()
def generate_gaussian_random_variables(mean, std_dev, size=1000):
    return [random.gauss(mean, std_dev) for _ in range(size)]
def gaussian_pdf(mean,std_dev,point):
    return (1/(std_dev*math.sqrt(2*math.pi)))*math.exp((-(point-mean)**2)/(2*std_dev**2))
def gaussian_cdf(mean,std_dev,point):
    return 0.5 * ((1 + math.erf((point - mean) / (std_dev * math.sqrt(2)))))
def gaussian(mean, std_dev,size):
    variance = std_dev**2
    histogram = generate_gaussian_random_variables(mean,std_dev,size)# generate the random variables for the histogram.
    points = line_space(mean-6*std_dev,mean+6*std_dev,100) #generates all the random variable values
    pdf = [gaussian_pdf(mean,std_dev,x)for x in points] # calculates thepdf for each point.
    cdf = [gaussian_cdf(mean,std_dev,x) for x in points] #calculates the cdf value for each point.
    # plots the PDF, CDF and histogram.
    plt.plot(points, pdf, label=f'µ: {mean}')
    plt.text(mean-10,0.4, f'var: {variance}', fontsize=12)
    plt.title('Gaussian PDF')
    plt.xlabel('X')
    plt.ylabel('probability')
    plt.legend()
    plt.show()
    plt.plot(points, cdf, label=f'µ: {mean}')
    plt.title('Gaussian CDF')
    plt.xlabel('X')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()
    plt.hist(histogram, bins='auto', edgecolor='black')
    plt.title("Gaussian histogram")
    plt.xlabel("X")
    plt.ylabel("samples")
    plt.show()
def calculate_mean(data):
    return sum(data) / len(data)
def calculate_variance(data, mean):
    return sum((x - mean) ** 2 for x in data) / len(data)
def calculate_standard_deviation(data):
    mean = calculate_mean(data)
    variance = calculate_variance(data, mean)
    return variance ** 0.5
def football_players_histogram():
    retirment_ages = [24,33,34,33,36,26,28,31,33,32,32,27,39,30,37,30,35,
                      34,31,29,36,35,33,32,35,32,35,35,34,32,36,31,33,33,29,32,33,35,
                      36,33,35,35,35,34,31,31,36,35,33,34,35,33,31,28,35,33,34,39,35,
                      41,35,37,40,36,32,36,36,34,31,35,37,39,36,37,36,36,35,36,36,35,
                      36,40,31,41,32,32,40,34,35,32,36,36,37,44,33,37,34,41,34,36,33,
                      35,35,29,32 ,31,30,27,47,35,35,37,38,36,45,44,45,30,28,34,39,32,
                      37,35,37,41,36,37,34,36,37,37,35,36,34,38,33,32,40,34,32,36,32,
                      38,35,35,35,41,30,35,37,31,36,38,39,34,39,37,43,41,38,35,32,36,
                      40,37,37,35,40,37,35,40,38,38,36,38,38,36,38,38,36,37,32,33,35,
                      37,33,38,36,33,39,36,38,35,38,35,39,41,41,37,35,35,36,35,39,37,
                      38,40,38,36,37,46,35]
    mean = calculate_mean(retirment_ages)
    std_dev = calculate_standard_deviation(retirment_ages)
    variance = calculate_variance(retirment_ages,mean)
    gaussian(mean,std_dev,1000)
    print("The mean:",mean)
    print("variance:",variance)
    print("Standard deviation:",std_dev)

print ("Choose which type of random variables that you will use:")
print ("1. Binomial")
print ("2. Bernoulli")
print ("3. Geometric")
print ("4. Poisson")
print ("5. Uniform Discrete")
print ("6. Continuous Uniform")
print ("7. Exponential")
print ("8. Gaussian")
print ("9. Football")

choice = input("Enter the corresponding number (1-8): ")
size = 1000

if choice == "1":
 n = int(input("Enter the number of trials (n): "))
 r = int(input("Enter the number of successful outcomes (r): "))
 p = float(input("Enter the probability of success (p): "))
 cdf = sum(binomial_pmf(n,i,p) for i in range(r + 1))
 binomial(n,p,size)
 print("The PMF of Binomial distribution: " , binomial_pmf(n,r,p))
 print("The CDF of Binomial distribution: " , cdf)
elif choice == "2":
 p = float(input("Enter the probability of success (p): "))
 n = int(input("Enter the number 0 or 1 (n): "))
 bernoulli(p,size)
 print("The PMF of Bernoulli distribution: " , bernoulli_pmf(p,n))
 print("The CDF of Bernoulli distribution: " , bernoulli_cdf(p,n))
elif choice == "3":
 p = float(input("Enter the probability of success (p): "))
 n = int(input("Enter the number of trials (n): "))
 geometric(p,size)
 print("The PMF of Geomatric distribution: " , geometric_pmf(p,n))
 print("The CDF of Geomatric distribution: " , geometric_cdf(p,n))
elif choice == "4":
 lambda_ = float(input("Enter lambda for Poisson distribution: "))
 k = int(input("Enter the number of events for Poisson distribution: "))
 a = int(input("Enter lower bound for Uniform distribution: "))
 b = int(input("Enter upper bound for Uniform distribution: "))
 poisson(lambda_,a,b,size)
 print("The PMF of Poisson distribution: " , poisson_pmf(lambda_,k))
 print("The CDF of Poisson distribution: " , poisson_cdf(lambda_,k))
elif choice == "5":
 a = int(input("Enter lower bound for Uniform distribution: "))
 b = int(input("Enter upper bound for Uniform distribution: "))
 n = int(input("Enter a point for the distribution: "))
 print("The PMF of discrete uniform distribution: " , discrete_uniform_pmf(a,b,n))
 print("The CDF of discrete uniform distribution: " , discrete_uniform_cdf(a,b,n))
 uniform_discrete(a,b,size)
elif choice == "6":
 a = int(input("Enter lower bound for Uniform distribution: "))
 b = int(input("Enter upper bound for Uniform distribution: "))
 n = int(input("Enter a point for the distribution: "))
 npoints = int(input("Enter the nunber of points for the distribution: "))
 continous_uniform(a,b,npoints,size)
 print("The PMF of continous uniform distribution: " , continous_uniform_pmf(a,b,n))
 print("The CDF of continous uniform distribution: " , continous_uniform_cdf(a,b,n))
elif choice == "7":
 lambda_ = float(input("Enter lambda for Exponential distribution: "))
 n = int(input("Enter a value at which to calculate the PDF: "))
 a = int(input("Enter lower bound for Uniform distribution: "))
 b = int(input("Enter upper bound for Uniform distribution: "))
 npoints = int(input("Enter the nunber of points for the distribution: "))
 exponential(lambda_,a,b,npoints,size)
 print("The PMF of Exponential distribution: " , exponential_pdf(lambda_,n))
 print("The CDF of Exponential distribution: " , exponential_cdf(lambda_,n))
elif choice == "8":
 m = float(input("Enter mean for Gaussian distribution: "))
 s = float(input("Enter standard deviation for the Gaussian distribution: "))
 n = int(input("Enter a value at which to calculate the PDF: "))
 gaussian(m,s,size)
 print("The PMF of Gaussian(Normal) distribution: " , gaussian_pdf(m,s,n))
 print("The CDF of Gaussian(Normal) distribution: " , gaussian_cdf(m,s,n))
elif choice == "9":
    football_players_histogram()
else:
 print("ERROR. Try again.")
 print("Please enter a number between 1 and 8.")