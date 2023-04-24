#!/usr/bin/perl                                                       

#######################################################################
# Obtained by modifying perl/master.pl                                # 
# Author: Himanshu Jain                                               # 
#######################################################################

#$time = 3600; #in seconds, timeout


$iterations = 1000;


$filename = "DATABASE.TXT";
open (READ_DATA, $filename);

$writelogs = $ARGV[0];
open (WRITEFILE, ">".$writelogs);


$line = <READ_DATA>;
$save = " ";

$command = " ";

while ($line ne "") {
    chop ($line);
    
    if ($line =~ /^[ ]*[\#]+/) {
	print "Ignoring comment \n".$line."\n";
    }
    else {

	$term_index = index($line, ";"); 

	if ( $term_index != -1) {
	    $line = substr($line,0,$term_index);
	}
    
	$command = $command." ".$line; 
    
	if ($term_index != -1) {
	    @args = split(/ +/, $command);

	    # Let us generate the command
	    $num_elements = @args;

	    if ( $num_elements < 3) {
		die "Each line should have a directory name, verilog file name, property name \n";
	    }

	    $dir = $args[1];
	    $vfile = $dir."/".$args[2]; 
	    $propfile = $dir."/".$args[3]; 
	
	    # some sanity checks

	    if (index($vfile, ".v") == -1) {
		die "Expected $vfile with .v extension \n";
	    }
	    
	    if (index($propfile, ".prop") == -1) {
		die "Expected $propfile with .prop extension \n";
	    }
	    
	    $command = "vcegar ".$vfile." --p ".$propfile." --i $iterations"; 
	    
	    # add remaining command line options
	    
	    for ($i = 4 ; $i < $num_elements ; $i ++) 
	    {
		$command = $command." ".$args[$i];
	    }
	    
	    print $command."\n\n";
	    
	    print WRITEFILE "#".$command."\n";
	    
	    @out = `$command`;
	    
	    for ($i = -23 ; $i < 0 ; $i++) { 
		print WRITEFILE $out[$i];
	    }
	    
	    print WRITEFILE "----------------------------------------\n\n";

	    print @out;

	    print "----------------------------------------\n\n";
	    
	    $command = "";
	}

    }
    $line = <READ_DATA>;
}

close (READ_DATA);
close (WRITEFILE);

#
#$filename = $ARGV[0]; #gcd.c
#$index1 = index($filename, ".c"); 
#$file = substr($filename,0,$index1); #gcd
#$tmpstr = $file."_*.cnf";
#@dirname = `ls -a $tmpstr`;
#
#$writefile= $file.".chaff";
#open (WRITEFILE, ">".$writefile);
#
#$successfile = ">"."$file."."chaff.success"; 
#open (SUCCESS, $successfile);
#
#$killed = ">"."$file."."chaff.killed"; 
#open (KILLED, $killed);
#
#foreach $file1 (@dirname) {
#    print "Handling $file1\n";
#    print "zchaff $file1\n";
#
#    $procid = fork();
#    if ($procid == 0) { #The kid
#	#@out = exec("zchaff $file1>tmp");
#	@out = `zchaff $file1`;
#	#print @out;
#	#print $out[-3];
#	#print $out[-1];
#	print WRITEFILE "zchaff $file1...\n";
#	print WRITEFILE $out[-3];
#	print WRITEFILE $out[-1];
#	print WRITEFILE "\n------------------\n";
#	print SUCCESS   "$file1 \n";
#	exit;
#    }
#    else { #I am a parent.
#	$SIG{ALRM} = sub { 
#	    print "...$file1 exceeded alotted time\n";
#	    print KILLED ("...$file1 exceeded alotted time\n");
#	    	  
#	    print WRITEFILE "zchaff $file1 killed...\n";
#	    kill (9, $procid);
#	    print ("procid is $procid");
#	};
#	alarm ($time);
#	wait;
#	#print @out[-1];
#	#print @out[-3];
#    }
#    
#    
#    @list = `ps -fu hjain`; 
#    foreach $line (@list) 
#    { 
#	if($line =~ /zchaff/) 
#	{ 
#	    print "Killing $line";
#	    #  if($line =~ /^\s*(\d+)\s+/) 
#	    #  { 
#	    #	$pid_ = $1; 
#	    #	print "Detected $pid is \n";
#		    #	kill 9, $pid_; 
#	    #    }
#	    chop($line);
#	    @b = split(/ +/, $line);
#	    print "killing $b[1]\n";
#	    kill 9, $b[1];
#	}
#    } 
#}
#
#close(WRITEFILE);
#close(SUCCESS);
#close(KILLED);

